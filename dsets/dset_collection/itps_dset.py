#!/usr/bin/env python3
import shutil
from fastai.data.all import untar_data, Path, get_files, progress_bar
from typing import Tuple
import pandas as pd
import os
from dset_config.dset_config import DatasetConfig

ANNOTATION_DATA_URL = (
    "https://www.dropbox.com/s/92tbhcdlymk5s0w/annotation_data.tar.gz?dl=1"
)


def _subset_data(df: pd.DataFrame, train_test):
    if train_test == "train":
        df = df[df["train"]].copy()
    elif train_test == "test":
        df = df[~df["train"]].copy()
    elif train_test == "both":
        pass
    return df


def _subset_lang(df: pd.DataFrame, lang=None):
    if lang is "both" or lang is None:
        return df
    elif lang == "en":
        return df[df["lang"] == lang].reset_index(drop=True)
    elif lang == "jp":
        return df[df["lang"] == lang].reset_index(drop=True)
    else:
        raise AttributeError("Unsupported language")


def get_annotation_data(
    dset: DatasetConfig, force_download=False
) -> Tuple[Path, pd.DataFrame]:
    p = untar_data(ANNOTATION_DATA_URL, force_download=force_download)
    if not isinstance(p, Path):
        raise AttributeError(f"Failed to unzip URL dataset under {ANNOTATION_DATA_URL}")
    df = pd.read_csv(p / "annotation_data.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise AttributeError("Failed to read csv after untaring dataset.")
    df = _subset_data(df, dset.split)
    if dset.lang is None:
        lang = "both"
    else:
        lang = dset.lang
    df = _subset_lang(df, lang)
    return p, df
