#!/usr/bin/env python3
import os
import tarfile
from pathlib import Path

import pandas as pd
from fastai.data.transforms import RandomSplitter


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def train_test_split(df: pd.DataFrame) -> pd.DataFrame:
    splits = RandomSplitter(seed=42, valid_pct=0.2)(df)
    df["filename"] = df["filename"].apply(Path)
    df["test"] = False
    df.loc[splits[1], "test"] = True
    return df
