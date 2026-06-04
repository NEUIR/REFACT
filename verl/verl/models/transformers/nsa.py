#!/usr/bin/env python
# encoding: utf-8

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from block_sparse_attn import (
    block_sparse_attn_func,
)
from einops import rearrange
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_flash_attention_utils import (
    _upad_input,
    # deterministic_g,
    fa_peft_integration_check,
    # flash_241,
    flash_attn_supports_top_left_mask,
    # prepare_fa2_from_position_ids,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    is_flash_attn_2_available,
    is_torch_flex_attn_available,
    is_torch_npu_available,
    logging,
)
from transformers.utils.import_utils import is_torch_fx_available

from verl.models.transformers.compressed_attention import compressed_attention

_use_top_left_mask = flash_attn_supports_top_left_mask()


logger = logging.get_logger(__name__)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


flash_attn_func = None

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb  # noqa


# patch functions in package `flash-attn` when using flash-attention on Ascend NPU.
if is_torch_npu_available():
    from torch_npu import npu_rotary_mul as apply_rotary_emb  # noqa

    from transformers.integrations.npu_flash_attention import (
        pad_input,
    )
    from transformers.integrations.npu_flash_attention import (
        npu_flash_attn_func as flash_attn_func,
    )
    from transformers.integrations.npu_flash_attention import (
        npu_flash_attn_varlen_func as flash_attn_varlen_func,
    )


if flash_attn_func:
    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )


# nsa
# debug token
debug_token = 3
token_now = 0
save_no_cache = False
save_cache = False


def convert_topk_to_base_blockmask(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    将topk索引转换为块稀疏注意力掩码，仅处理-1的情况

    Args:
        topk_idx: 形状 [num_heads, total_seqlen, k] 的块索引张量
        cu_seqlens_q: 累积序列长度（用于计算总长度）
        max_seqlen_k: 最大键序列长度（用于计算键块数量）
        block_size: block_size
        device: 输出设备

    Returns:
        mask: 布尔掩码，形状 [num_heads, total_seqlen, k_blocks]
    """
    # 计算键块数量
    k_blocks = (max_seqlen_k + block_size - 1) // block_size  # 向上取整
    num_heads, total_seqlen, k = topk_idx.shape

    # 初始化全False掩码
    mask = torch.zeros(
        num_heads, total_seqlen, k_blocks, dtype=torch.bool, device=device
    )

    # 过滤掉 -1，确保索引合法
    valid_idx = topk_idx[topk_idx != -1]

    # 生成索引掩码
    batch_idx, seq_idx, _ = torch.where(
        topk_idx != -1
    )  # 找到非-1索引的 (head, seq) 位置
    mask[batch_idx, seq_idx, valid_idx] = True  # 设置对应位置为 True

    return mask


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, moba_chunk_size, kernel_stride):
    """
    计算需要 MOBA 注意力的 chunks，支持 stride。
    返回:
        - filtered_indices: 用于直接索引 kv 的索引。
        - cu_seqlens_compressed: 压缩后的累积序列长度。
    """
    # 1. 计算每个序列的长度
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 2. 计算每个序列的 chunk 起始位置 (考虑 stride)
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (
        max_seq_len - moba_chunk_size
    ) // kernel_stride + 1  # 修正公式
    chunk_start_offsets = torch.arange(
        0,
        max_num_chunks_per_seq * kernel_stride,
        kernel_stride,
        device=cu_seqlen.device,
    )
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = (
        seq_starts[:, None] + chunk_start_offsets[None, :]
    )  # [batch_size, max_num_chunks_per_seq]

    # 3. 过滤掉超出序列长度的 chunk 和非完整大小的 chunk
    chunk_end_in_seq = chunk_start_in_seq + moba_chunk_size
    valid_chunk_mask = chunk_end_in_seq <= (
        seq_starts[:, None] + batch_sizes[:, None]
    )  # 完整 chunk

    # 4. 根据 valid_chunk_mask 过滤有效的 chunk 起始位置
    valid_chunk_starts = chunk_start_in_seq[
        valid_chunk_mask
    ]  # [num_valid_chunks]
    del chunk_start_in_seq
    # 5. 生成 filtered_indices
    chunk_indices = torch.arange(0, moba_chunk_size, device=cu_seqlen.device)[
        None, :
    ]  # [1, moba_chunk_size]
    filtered_indices = (
        valid_chunk_starts[:, None] + chunk_indices
    )  # [num_valid_chunks, moba_chunk_size]
    filtered_indices = filtered_indices.view(-1)  # 展平为一维索引

    # 6. 计算压缩后的累积序列长度
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(
        dim=1
    )  # 每个 batch 的有效 chunk 数量
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    del (
        num_filtered_chunks_per_batch,
        chunk_start_offsets,
        seq_starts,
        chunk_end_in_seq,
        valid_chunk_mask,
        chunk_indices,
    )
    return filtered_indices, cu_seqlens_compressed


class CompressKV(torch.nn.Module):
    def __init__(
        self,
        head_num_k,
        head_dim,
        kernel_size,
        compress_func,
        add_pos_embed=False,
        kernel_stride=16,
    ):
        """
        压缩KV模块，支持多种压缩方式
        Args:
            head_num_k: KV头的数量
            head_dim: 每个头的维度
            kernel_size: 每个chunk的大小
            compress_func: 压缩方式（如meanpool, mlp, conv1d等）
            add_pos_embed: 是否添加位置编码
            stride: 分块时的步长
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.compress_func = compress_func
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride  # 新增stride参数

        # 定义不同的压缩方式
        if compress_func == "mlp" or compress_func == "mlp+residual":
            self.kv_compress = nn.Sequential(
                nn.Linear(kernel_size * 2, kernel_size * 4),
                nn.ReLU(),
                nn.Linear(kernel_size * 4, 2),
            )
        elif compress_func == "conv1d":
            self.k_conv = nn.Conv1d(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                kernel_size=kernel_size,
            )
            self.v_conv = nn.Conv1d(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                kernel_size=kernel_size,
            )
        elif compress_func == "weighted_sum":
            self.weight_net_v = nn.Linear(self.head_dim, 1)
            self.weight_net_k = nn.Linear(self.head_dim, 1)
        elif compress_func == "weighted_sum+proj":
            self.weight_net_v = nn.Linear(self.head_dim, 1)
            self.weight_net_k = nn.Linear(self.head_dim, 1)
            self.k_proj = nn.Linear(self.head_dim, self.head_dim)
            self.v_proj = nn.Linear(self.head_dim, self.head_dim)

        if add_pos_embed:
            # 修改位置编码层：为每个头创建独立的位置编码
            self.pos_embed = nn.Embedding(
                kernel_size,
                head_num_k
                * head_dim,  # 维度扩展为 [kernel_size, num_heads * head_dim]
            )
        else:
            self.pos_embed = None

    def forward(self, kv: torch.Tensor, cu_seqlens):
        """
        前向传播，压缩KV
        Args:
            kv: 输入的KV张量
            cu_seqlens: 累积序列长度
        Returns:
            compress_k: 压缩后的K
            compress_v: 压缩后的V
            cu_seqlens_compressed: 压缩后的累积序列长度
        """

        # 计算chunk相关信息，支持stride
        filtered_kv_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )

        # 提取过滤后的kv
        filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

        # 分块
        filtered_kv = filtered_kv.view(
            filtered_kv.shape[0] // self.kernel_size,
            self.kernel_size,
            2,
            self.head_num_k,
            self.head_dim,
        )  # [l, block_size,2,h,d]
        if self.pos_embed is not None:
            positions = torch.arange(self.kernel_size, device=kv.device)
            pos_emb = self.pos_embed(
                positions
            )  # [kernel_size, num_heads * head_dim]

            # 重塑形状以匹配多头结构
            pos_emb = pos_emb.view(
                self.kernel_size,
                self.head_num_k,  # 使用实际头数参数（需在__init__中保存）
                self.head_dim,
            )  # [kernel_size, num_heads, head_dim]

            # 添加维度用于广播
            pos_emb = pos_emb.reshape(
                1, self.kernel_size, 1, self.head_num_k, self.head_dim
            )  # [1, block_size, 1, num_heads, head_dim]
            filtered_kv = filtered_kv + pos_emb

        if self.compress_func == "meanpool":
            compressed_kv = filtered_kv.mean(dim=1)
            compress_k = compressed_kv[
                :, 0, :, :
            ]  # .reshape(-1, self.head_num_k, self.head_dim)
            compress_v = compressed_kv[
                :, 1, :, :
            ]  # .reshape(-1, self.head_num_k, self.head_dim)
        elif self.compress_func == "mlp":
            filtered_kv = filtered_kv.permute(0, 3, 4, 2, 1).reshape(
                filtered_kv.shape[0], self.head_num_k, self.head_dim, -1
            )
            compressed_kv = self.kv_compress(filtered_kv)
            compress_k = compressed_kv[
                :, :, :, 0
            ]  # .reshape(-1, self.head_num_k, self.head_dim)
            compress_v = compressed_kv[
                :, :, :, 1
            ]  # .reshape(-1, self.head_num_k, self.head_dim)
        elif self.compress_func == "mlp+residual":
            mean_kv = filtered_kv.mean(dim=1)
            mlp_kv = self.kv_compress(
                filtered_kv.permute(0, 3, 4, 2, 1).reshape(
                    filtered_kv.shape[0], self.head_num_k, self.head_dim, -1
                )
            ).permute(0, 3, 1, 2)  # [l, h,d,2]->[l,2,h,d]
            compressed_kv = mean_kv + mlp_kv
            compress_k = compressed_kv[:, 0, :, :]
            compress_v = compressed_kv[:, 1, :, :]
        elif self.compress_func == "conv1d":
            k = filtered_kv[:, :, 0, :, :]
            k = rearrange(
                k, "l block_size h d -> (l h) d block_size"
            )  # 只能3维
            v = filtered_kv[:, :, 1, :, :]
            v = rearrange(v, "l block_size h d -> (l h) d block_size")
            compress_k = self.k_conv(k).squeeze(-1)  # [(l h), d]
            compress_v = self.v_conv(v).squeeze(-1)  # [(l h), d]
            compress_k = rearrange(
                compress_k, "(l h) d -> l h d", h=self.head_num_k
            )
            compress_v = rearrange(
                compress_v, "(l h) d -> l h d", h=self.head_num_k
            )

        elif self.compress_func == "weighted_sum":
            k = filtered_kv[:, :, 0, :, :]
            k = rearrange(k, "l block_size h d -> l h block_size d")
            v = filtered_kv[:, :, 1, :, :]
            v = rearrange(v, "l block_size h d -> l h block_size d")
            weight_k = torch.softmax(
                self.weight_net_k(k), dim=2
            )  # [l, h, block_size, 1]
            weight_v = torch.softmax(
                self.weight_net_v(v), dim=2
            )  # [l, h, block_size, 1]

            compress_k = (weight_k * k).sum(dim=2)  # [l, h, d]
            compress_v = (weight_v * v).sum(dim=2)  # [l, h, d]
        elif self.compress_func == "weighted_sum+proj":
            k = filtered_kv[:, :, 0, :, :]
            k = rearrange(k, "l block_size h d -> l h block_size d")
            v = filtered_kv[:, :, 1, :, :]
            v = rearrange(v, "l block_size h d -> l h block_size d")
            weight_k = torch.softmax(
                self.weight_net_k(k), dim=2
            )  # [l, h, block_size, 1]
            weight_v = torch.softmax(
                self.weight_net_v(v), dim=2
            )  # [l, h, block_size, 1]

            compress_k = (weight_k * self.k_proj(k)).sum(dim=2)  # [l, h, d]
            compress_v = (weight_v * self.v_proj(v)).sum(dim=2)  # [l, h, d]

        else:
            raise ValueError(f"Unsupported compress type: {self.compress_func}")

        del filtered_kv
        if "compressed_kv" in locals():
            del compressed_kv

        return compress_k, compress_v, cu_seqlens_compressed


class DynamicCacheQKV(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        if num_hidden_layers is None:
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
            self.query_cache: List[torch.Tensor] = []
        else:
            self.key_cache: List[torch.Tensor] = [
                [] for _ in range(num_hidden_layers)
            ]
            self.value_cache: List[torch.Tensor] = [
                [] for _ in range(num_hidden_layers)
            ]
            self.query_cache: List[torch.Tensor] = [
                [] for _ in range(num_hidden_layers)
            ]
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                self.query_cache[layer_idx],
            )
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                self.query_cache[layer_idx],
            )

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        query_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if query_states is None:
            raise ValueError(
                "query_states must be provided for DynamicCacheQKV"
            )

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.query_cache.append(query_states)
        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.query_cache[layer_idx] = query_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
            self.query_cache[layer_idx] = torch.cat(
                [self.query_cache[layer_idx], query_states], dim=-2
            )

        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            self.query_cache[layer_idx],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx or (
            len(self.key_cache) > layer_idx and self.key_cache[layer_idx] == []
        ):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(
        self,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += (
                (self.key_cache[layer_idx], self.value_cache[layer_idx]),
            )
        return legacy_cache

    # @classmethod
    # def from_legacy_cache(
    #     cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_hidden_layers: int = None
    # ) -> "DynamicCacheQKV":
    #     """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
    #     backward compatibility."""
    #     cache = cls(num_hidden_layers)
    #     if past_key_values is not None:
    #         for layer_idx in range(len(past_key_values)):
    #             key_states, value_states, query_status = past_key_values[layer_idx]
    #             cache.update(key_states, value_states, query_status,layer_idx)
    #     return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][
                    ..., :max_length, :
                ]
                self.query_cache[idx] = self.query_cache[idx][
                    ..., :max_length, :
                ]

    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int
    ) -> List["DynamicCacheQKV"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCacheQKV(num_hidden_layers)
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [
                tensor[i : i + split_size] for tensor in self.key_cache
            ]
            current_split.value_cache = [
                tensor[i : i + split_size] for tensor in self.value_cache
            ]
            current_split.query_cache = [
                tensor[i : i + split_size] for tensor in self.query_cache
            ]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(
        cls, splits: List["DynamicCacheQKV"], num_hidden_layers: int
    ) -> "DynamicCacheQKV":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls(num_hidden_layers)
        for idx in range(len(splits[0])):
            key_cache = [
                current.key_cache[idx]
                for current in splits
                if current.key_cache[idx] != []
            ]
            value_cache = [
                current.key_cache[idx]
                for current in splits
                if current.key_cache[idx] != []
            ]
            query_cache = [
                current.key_cache[idx]
                for current in splits
                if current.key_cache[idx] != []
            ]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                layer_query = torch.cat(query_cache, dim=0)
                cache.update(
                    layer_keys, layer_values, idx, query_states=layer_query
                )
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[
                layer_idx
            ].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[
                layer_idx
            ].repeat_interleave(repeats, dim=0)
            self.query_cache[layer_idx] = self.query_cache[
                layer_idx
            ].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][
                indices, ...
            ]
            self.query_cache[layer_idx] = self.query_cache[layer_idx][
                indices, ...
            ]


# nsa
# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(
        _prepare_4d_causal_attention_mask
    )


def _nsa_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    # print(f"verl _nsa_forward position_ids: {position_ids}, max_length_q: {max_length_q}, query_length: {query_length}")
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size
        and sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    flash_kwargs = (
        {"window_size": (sliding_window, sliding_window)}
        if use_sliding_windows
        else {}
    )

    # if flash_241:
    #     if deterministic is None:
    #         deterministic = deterministic_g
    #     flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = _upad_input(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = nsa_varlen_func(
            module,
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_in_batch_q=max_seqlen_in_batch_q,
            max_seqlen_in_batch_k=max_seqlen_in_batch_k,
            # dropout_p=dropout,
            # softmax_scale=softmax_scale,
            # causal=causal,
            # **flash_kwargs,
        )
        attn_output = pad_input(
            attn_output_unpad, indices_q, batch_size, query_length
        )

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    elif position_ids is not None:
    #elif position_ids is not None and (
    #    max_length_q is not None
    #    or (
    #        query_length != 1
    #        and not (torch.diff(position_ids, dim=-1) >= 0).all()
    #    )
    #):
        batch_size = query_states.size(0)

        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            # (
            #     query_states,
            #     key_states,
            #     value_states,
            #     indices_q,
            #     cu_seq_lens,
            #     max_seq_lens,
            # ) = prepare_fa2_from_position_ids(
            #     query_states, key_states, value_states, position_ids
            # )

            cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
            max_length_q, max_length_k = max_seq_lens

        else:
            query_states = query_states.reshape(
                -1, query_states.size(-2), query_states.size(-1)
            )
            key_states = key_states.reshape(
                -1, key_states.size(-2), key_states.size(-1)
            )
            value_states = value_states.reshape(
                -1, value_states.size(-2), value_states.size(-1)
            )

        # print(f"verl nsa, before nsa varlen: q: {query_states.shape}, k: {key_states.shape}, v: {value_states.shape}, position_ids: {position_ids}")
        attn_output = nsa_varlen_func(
            module,
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_in_batch_q=max_length_q,
            max_seqlen_in_batch_k=max_length_k,
            # dropout_p=dropout,
            # softmax_scale=softmax_scale,
            # causal=causal,
            # **flash_kwargs,
        )

        attn_output = attn_output.view(
            batch_size, -1, attn_output.size(-2), attn_output.size(-1)
        )

    else:
        raise ValueError
    # print("verl nsa forward ended")

    return attn_output


def nsa_varlen_func(
    self,
    query_layer,
    key_layer,
    value_layer,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_in_batch_q,
    max_seqlen_in_batch_k,
    original_hidden_states=None,
):
    kv = torch.stack((key_layer, value_layer), dim=1)
    compressed_k, compressed_v, compressed_cu_seqlens = self.compress_kv(
        kv, cu_seqlens_k
    )
    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    compressed_attn_output, topk_idx = compressed_attention(
        query_layer,
        compressed_k,
        compressed_v,
        self.kernel_size,
        self.kernel_stride,
        self.block_size,
        self.topk,
        cu_seqlens_q,
        compressed_cu_seqlens,
        max_seqlen_in_batch_q,
        compressed_seqlens.max().item(),
        None,
        init_blocks=self.init_blocks,
        local_blocks=self.local_blocks,
    )

    del (
        compressed_k,
        compressed_v,
        compressed_cu_seqlens,
        kv,
        compressed_seqlens,
    )
    nheads_k = key_layer.shape[1]
    head_mask_type = torch.tensor(
        [1] * nheads_k, device=query_layer.device, dtype=torch.int32
    )
    streaming_info = torch.tensor(
        [0, 0] * nheads_k, device=query_layer.device, dtype=torch.int32
    )
    exact_streaming = False

    repeat_times = 1
    if repeat_times > 1:
        query_layer_repeat = query_layer.repeat_interleave(repeat_times, dim=-2)
    else:
        query_layer_repeat = query_layer
    topk_attn_output = block_sparse_attn_func(
        query_layer_repeat,
        key_layer,
        value_layer,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        topk_idx,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        self.attention_dropout,
        deterministic=False,
        softmax_scale=None,
        is_causal=True,
        exact_streaming=False,
        return_attn_probs=False,
        block_window_size=self.window_size // self.block_size,
        use_checkpoint=False,
    )
    # import pdb; pdb.set_trace()
    # raise ValueError('debug')
    if repeat_times > 1:
        topk_attn_output = topk_attn_output.view(
            topk_attn_output.shape[0],
            topk_attn_output.shape[1] // repeat_times,
            repeat_times,
            -1,
        ).mean(dim=-2)
    return topk_attn_output
