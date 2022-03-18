#!/usr/bin/env python3
from fastai.data.all import untar_data
import pandas as pd
from .dset_config import DatasetConfig

LJ_SPEECH_URL = "https://www.dropbox.com/s/h0d8fa13ylwpssq/LJSpeech-1.1.tar.gz?dl=1"

def get_ljl_data(dset_config: DatasetConfig):
    p = untar_data(LJ_SPEECH_URL)
    df = pd.read_csv(p / "metadata.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    if dset_config.split == "train": df = df[~df["test"]].copy()
    elif dset_config.split == "test": df = df[df["test"]].copy()

    return p, df
