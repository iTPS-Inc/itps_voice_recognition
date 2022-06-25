#!/usr/bin/env python3
#!/usr/bin/env python3
import pandas as pd
from dsets.dset_config.dset_config import DatasetConfig
from fastai.data.all import untar_data, Path
from typing import Tuple

LJ_SPEECH_URL = "https://www.dropbox.com/s/h0d8fa13ylwpssq/LJSpeech-1.1.tar.gz?dl=1"
JSUT_URL = "https://www.dropbox.com/s/o949otj06b9ucmm/jsut_ver1.1.tar.gz?dl=1"
SPREDS_URL = "https://www.dropbox.com/s/7325sa83zdgl3le/NICT_SPREDS.tar.gz?dl=1"
TEST_DATA_URL = "https://www.dropbox.com/s/hrfmsjadepupiwu/test_data.tar.gz?dl=1"


def get_ljl_data(dset_config: DatasetConfig, base=None, force_download=False):
    p = untar_data(LJ_SPEECH_URL, force_download=force_download)
    df = pd.read_csv(p / "metadata.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    if dset_config.split == "train":
        df = df[~df["test"]].copy()
    elif dset_config.split == "test":
        df = df[df["test"]].copy()
    df["filename"] = df["filename"].apply(lambda x: p / x)
    return p, df


def get_jsut_data(dset_config: DatasetConfig, base=None, force_download=False):
    p = untar_data(JSUT_URL, force_download=force_download)
    df = pd.read_csv(p / "metadata.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    if dset_config.split == "train":
        df = df[~df["test"]].copy()
    elif dset_config.split == "test":
        df = df[df["test"]].copy()
    df["filename"] = df["filename"].apply(lambda x: p / x)
    return p, df


def get_nictspreds_data(dset_config: DatasetConfig, base=None, force_download=False):
    p = untar_data(SPREDS_URL, force_download=force_download)
    df = pd.read_csv(p / "metadata.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    df["filename"] = df["filename"].apply(lambda x: p / x)
    if dset_config.split == "train":
        df = df[~df["test"]].copy()
    elif dset_config.split == "test":
        df = df[df["test"]].copy()
    if dset_config.lang == "all":
        pass
    else:
        df = df[df["language"] == dset_config.lang].copy()
    return p, df


def get_test_data(
    dset_config: DatasetConfig = DatasetConfig(name="test_data", split="both"),
    base=None,
    force_download=False,
) -> Tuple[Path, pd.DataFrame]:
    p = untar_data(TEST_DATA_URL, force_download=force_download)
    df = pd.read_csv(p / "metadata.csv")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Couldn't read dataframe correctly")
    df["filename"] = df["filename"].apply(lambda x: p / x)
    if dset_config.split == "train":
        df = df[~df["test"]].copy()
    elif dset_config.split == "test":
        df = df[df["test"]].copy()
    return p, df
