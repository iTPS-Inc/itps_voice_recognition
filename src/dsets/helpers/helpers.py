#!/usr/bin/env python3
import os
import tarfile
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
from fastai.data.transforms import RandomSplitter
from tqdm import tqdm


def process_parallel(df_or_ser, fn, num_cores):
    with Pool(num_cores) as p:
        splitted_df = np.array_split(df_or_ser, num_cores)
        split_df_results = p.map(fn, splitted_df)
    df_or_ser = pd.concat(split_df_results)
    return df_or_ser


def apply_parallel(df_or_sersr, fn, num_cores):
    def _inner(df_or_ser):
        return df_or_ser.apply(fn)

    with ThreadPool(num_cores) as p:
        splitted_df = np.array_split(df_or_sersr, num_cores)
        split_df_results = list(
            tqdm(p.imap(_inner, splitted_df), total=len(splitted_df))
        )
    df_or_sersr = pd.concat(split_df_results)
    return df_or_sersr


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def train_test_split(df: pd.DataFrame) -> pd.DataFrame:
    splits = RandomSplitter(seed=42, valid_pct=0.2)(df)
    df["filename"] = df["filename"].apply(Path)
    df["test"] = False
    df.loc[splits[1], "test"] = True
    return df


def _get_srs(fn):
    _, sr = torchaudio.load(fn)
    return sr


def get_sampling_rates(fns, num_cores):
    sr = apply_parallel(fns, _get_srs, num_cores)
    return sr
