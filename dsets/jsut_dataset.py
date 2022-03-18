#!/usr/bin/env python3
from fastai.data.all import untar_data
import pandas as pd
from .dset_config import DatasetConfig

JSUT_URL = "https://www.dropbox.com/s/o949otj06b9ucmm/jsut_ver1.1.tar.gz?dl=1"

def get_jsut_data(dset_config: DatasetConfig):
    p = untar_data(JSUT_URL, force_download=True)
    df = pd.read_csv(p / "metadata.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    if dset_config.split == "train": df = df[~df["test"]].copy()
    elif dset_config.split == "test": df = df[df["test"]].copy()
    return p, df
