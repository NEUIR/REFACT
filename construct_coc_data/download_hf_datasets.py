from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# 预设：与 Hugging Face Hub 上的常用仓库对应
PRESETS = {
    "hotpot_care": {
        "repo_id": "sheryc/hotpotqa_care",
        "config": None,
        "split": "train",
        "description": "HotpotQA CARE（EMNLP 2025，含 reasoning / supporting_facts 等）",
    },
    "musique": {
        "repo_id": "bdsaglam/musique",
        "config": "default",
        "split": "train",
        "description": "MuSiQue（含 paragraphs，default=full 风格）",
    },
}


def _row_to_python(row: Any) -> dict:
    """将 datasets 的一行转为可 JSON 序列化的 dict。"""
    if hasattr(row, "items"):
        return {k: _value_to_python(v) for k, v in row.items()}
    return dict(row)


def _value_to_python(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [_value_to_python(x) for x in v]
    if isinstance(v, dict):
        return {k: _value_to_python(x) for k, v in v.items()}
    # numpy / pyarrow 标量
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def load_hf_split(repo_id: str, config: str | None, split: str, token: str | None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": repo_id, "split": split}
    if config:
        kwargs["name"] = config
    if token:
        kwargs["token"] = token
    return load_dataset(**kwargs)


def iter_rows(repo_id: str, config: str | None, split: str, token: str | None) -> Iterable[dict]:
    ds = load_hf_split(repo_id, config, split, token)
    n = len(ds)
    for i, row in enumerate(ds):
        if (i + 1) % 5000 == 0 or i + 1 == n:
            print(f"  已读取 {i + 1}/{n} ...", flush=True)
        yield _row_to_python(row)


def save_jsonl(rows: Iterable[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def save_json_array(rows: Iterable[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return len(data)


def download_one(
    name: str,
    repo_id: str,
    config: str | None,
    split: str,
    output_path: Path,
    fmt: str,
    token: str | None,
) -> int:
    print(f"\n[{name}] {PRESETS.get(name, {}).get('description', repo_id)}")
    print(f"  Hub: {repo_id}" + (f" (config={config})" if config else "") + f" split={split}")
    print(f"  输出: {output_path}")

    rows = iter_rows(repo_id, config, split, token)
    if fmt == "json":
        count = save_json_array(rows, output_path)
    else:
        count = save_jsonl(rows, output_path)

    print(f"  完成: {count} 条 -> {output_path}")
    return count


def list_musique_configs(repo_id: str, token: str | None) -> None:
    from datasets import get_dataset_config_names

    kwargs: dict[str, Any] = {"path": repo_id}
    if token:
        kwargs["token"] = token
    configs = get_dataset_config_names(**kwargs)
    print(f"可用 config: {configs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载 musique 与 hotpot_care 数据集")
    parser.add_argument(
        "--only",
        choices=["hotpot_care", "musique", "all"],
        default="all",
        help="只下载指定数据集（默认 all）",
    )

    # HotpotQA CARE
    parser.add_argument(
        "--hotpot-repo",
        default=PRESETS["hotpot_care"]["repo_id"],
        help="Hotpot CARE 的 HF 仓库 id",
    )
    parser.add_argument(
        "--hotpot-split",
        default=PRESETS["hotpot_care"]["split"],
        help="Hotpot CARE 划分（默认 train）",
    )
    parser.add_argument(
        "--hotpot-output",
        type=Path,
        default=Path("/user/jinzhensheng/hotpotqa/hotpotqa_care.jsonl"),
        help="Hotpot CARE 输出路径",
    )

    # MuSiQue
    parser.add_argument(
        "--musique-repo",
        default=PRESETS["musique"]["repo_id"],
        help="MuSiQue 的 HF 仓库 id",
    )
    parser.add_argument(
        "--musique-config",
        default=PRESETS["musique"]["config"],
        help="MuSiQue 子集：default（full）或 answerable",
    )
    parser.add_argument(
        "--musique-split",
        default=PRESETS["musique"]["split"],
        help="MuSiQue 划分：train / validation",
    )
    parser.add_argument(
        "--musique-output",
        type=Path,
        default=Path("/user/jinzhensheng/Musique/musique_full_v1.0_train.jsonl"),
        help="MuSiQue 输出路径",
    )

    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="保存格式：jsonl（每行一条）或 json（整个数组）",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token（私有/ gated 数据集时需要）",
    )
    parser.add_argument(
        "--list-musique-configs",
        action="store_true",
        help="列出 musique 仓库下所有 config 后退出",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print("请先安装依赖: pip install datasets huggingface_hub", file=sys.stderr)
        sys.exit(1)

    if args.list_musique_configs:
        list_musique_configs(args.musique_repo, args.hf_token)
        return

    targets: list[tuple[str, str, str | None, str, Path]] = []

    if args.only in ("all", "hotpot_care"):
        targets.append(
            (
                "hotpot_care",
                args.hotpot_repo,
                None,
                args.hotpot_split,
                args.hotpot_output,
            )
        )
    if args.only in ("all", "musique"):
        targets.append(
            (
                "musique",
                args.musique_repo,
                args.musique_config,
                args.musique_split,
                args.musique_output,
            )
        )

    total = 0
    for name, repo_id, config, split, out_path in targets:
        try:
            n = download_one(
                name, repo_id, config, split, out_path, args.format, args.hf_token
            )
            total += n
        except Exception as e:
            print(f"\n[{name}] 下载失败: {e}", file=sys.stderr)
            print(
                "提示: 检查网络、HF 仓库名、config/split 是否正确；"
                "gated 数据集需加 --hf-token",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"\n全部完成，共写入 {total} 条。")


if __name__ == "__main__":
    main()
