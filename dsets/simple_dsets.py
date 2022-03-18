#!/usr/bin/env python3
#!/usr/bin/env python3
from fastai.data.all import untar_data
import pandas as pd
from .dset_config import DatasetConfig

LJ_SPEECH_URL = "https://www.dropbox.com/s/h0d8fa13ylwpssq/LJSpeech-1.1.tar.gz?dl=1"
JSUT_URL = "https://www.dropbox.com/s/o949otj06b9ucmm/jsut_ver1.1.tar.gz?dl=1"

def get_ljl_data(dset_config: DatasetConfig, force_download=False):
    p = untar_data(LJ_SPEECH_URL, force_download=force_download)
    df = pd.read_csv(p / "metadata.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    if dset_config.split == "train": df = df[~df["test"]].copy()
    elif dset_config.split == "test": df = df[df["test"]].copy()
    return p, df


def get_jsut_data(dset_config: DatasetConfig, force_download=False):
    p = untar_data(JSUT_URL, force_download=force_download)
    df = pd.read_csv(p / "metadata.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    if dset_config.split == "train": df = df[~df["test"]].copy()
    elif dset_config.split == "test": df = df[df["test"]].copy()
    return p, df

def get_nictspreads_data(dset_config: DatasetConfig, force_download=False):

    p = untar_data(SPREADS_URL, force_download=force_download)
    df = pd.read_csv(p / "metadata.csv", index_col=0)
    langs = set(df["language"].unique())
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    if dset_config.split == "train": df = df[~df["test"]].copy()
    elif dset_config.split == "test": df = df[df["test"]].copy()
    if dset_config.lang == "all":
        pass
    else:
        df = df[df["lang"] == dset_config.lang].copy()
    return p, df
