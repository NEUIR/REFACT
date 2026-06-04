#!/usr/bin/env python
# encoding: utf-8

import json
import os

import fire
import polars as pl
import ray
from ray.experimental import tqdm_ray

from verl.utils.reward_score import _default_compute_score


@ray.remote
def process_row(row, bar: tqdm_ray.tqdm):
    scores = []
    for resp in row["responses"]:
        data_source = row["data_source"]
        ground_truth = row["reward_model"]["ground_truth"]
        extra_info = row.get("extra_info", {})
        score = _default_compute_score(
            data_source, resp, ground_truth, extra_info
        )
        scores.append(score)
    bar.update.remote(1)
    return sum(scores) / (len(scores) + 1e-6)


def batch_eval(data_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    hard_file = os.path.join(save_path, "data.parquet")

    # 读取 Parquet 文件
    pl_df = pl.read_parquet(data_path)
    # pl_df = pl_df.head(100)  # 限制行数用于测试
    pl_df = pl_df.with_row_count(name="index")

    # 初始化进度条
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    bar = remote_tqdm.remote(total=pl_df.height)

    # 使用 Polars 的 rows(named=True) 遍历行
    scores = ray.get(
        [process_row.remote(row, bar) for row in pl_df.iter_rows(named=True)]
    )

    # 关闭进度条
    ray.get(bar.close.remote())

    # 添加 pass_rate 列
    new_df = pl_df.with_columns(pass_rate=pl.Series(scores))

    # 保存到 Parquet
    new_df.write_parquet(hard_file)

    # 验证
    print(pl.read_parquet(hard_file))


if __name__ == "__main__":
    fire.Fire(batch_eval)
