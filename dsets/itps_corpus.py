#!/usr/bin/env python3
from ctypes import ArgumentError
import shutil
from fastai.data.all import untar_data, Path, get_files, progress_bar
from typing import Tuple
from fastprogress.fastprogress import ProgressBar
import pandas as pd
import os

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


def _subset_lang(df: pd.DataFrame, lang: str):
    if lang is "both":
        return df
    elif lang == "en":
        return df[df["lang"] == lang].reset_index(drop=True)
    elif lang == "jp":
        return df[df["lang"] == lang].reset_index(drop=True)
    else:
        raise ArgumentError("Unsupported language")


def get_annotation_data(lang="both", train_test="train") -> Tuple[Path, pd.DataFrame]:
    p = untar_data(ANNOTATION_DATA_URL)
    if not isinstance(p, Path):
        raise ArgumentError(f"Failed to unzip URL dataset under {ANNOTATION_DATA_URL}")
    df = pd.read_csv(p / "annotation_data.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise ArgumentError("Failed to read csv after untaring dataset.")
    df = _subset_data(df, train_test)
    df = _subset_lang(df, lang)
    return p, df
