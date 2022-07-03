#!/usr/bin/env python3
import pandas as pd
from dsets.dset_config.dset_config import DatasetConfig
from fastai.data.all import untar_data, Path
from typing import Tuple
from fastdownload import FastDownload
import unicodedata
from tqdm import tqdm
import string
import re

tqdm.pandas()

OTHER_DATA_URL = "https://www.dropbox.com/s/5lqvn6gkte73vbm/other_data.tar.gz?dl=1"

CHARACTERS = (
    "a-zA-Z"
    + "\u3040-\u309F"  # Alphabet
    + "\u30A0-\u30FF"  # Hiragana
    + "\u4E00-\u9FAF"  # Katakana
    + "\u3400-\u4DB5"  # Kanji
    + "\u4E00-\u9FCB"  # Extended Kanji
    + "\uF900-\uFA6A"  # Extended Kanji  # Extended Kanji
)
CHARS = re.compile(f"[{CHARACTERS}]+")


def super_clean(s):
    s = unicodedata.normalize("NFKC", s)  # Normalize
    s = re.sub(r"(\(.*\))", "", s)  # Replace text in parens
    s = re.sub(r"\s+", "", s)  # No spaces, since mecab does it
    s = re.sub(
        "[^"
        + CHARACTERS
        + "0-9"
        + "\u3000-\u3002"
        + "\u30FC"  # 　、。
        + "\u300C\u300D"  # ー
        + "\uFF08\uFF09"  # ｢｣
        + "\uFF01\uFF1F"  # （）
        + "\u3000-\u303F"  # Full width punctuation, numbers, latin
        + "\uFF01-\uFF60"  # Japanese symbols, punctuation
        + string.punctuation  # Full-width sumbols and punctuation！？
        + "]",
        "",
        s,
    )
    if not CHARS.findall(s):
        return "[NO SPEECH]"
    return s


def regular_clean(s):
    s = unicodedata.normalize("NFKC", s)  # Normalize
    s = re.sub(r"\s+", "", s)  # No spaces, since mecab does it
    return s


def _subset_data(df: pd.DataFrame, train_test):
    if train_test == "train":
        df = df[df["train"]].copy()
    elif train_test == "test":
        df = df[~df["train"]].copy()
    elif train_test == "both":
        pass
    return df


def get_other_data(dset: DatasetConfig, base="~/.fastdownload", force_download=False):
    p = untar_data(OTHER_DATA_URL, force_download=force_download)
    df = pd.read_csv(p / "clip_frame.csv", index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise AttributeError("Failed to read csv after untaring dataset.")
    df = prep_other_data(df, dset)
    return p, df


def prep_other_data(df: pd.DataFrame, dset: DatasetConfig):
    df = _subset_data(df, dset.split)
    df["super_clean"] = df["content"].apply(super_clean)
    df["kinda_clean"] = df["content"].apply(regular_clean)
    df["file"] = df["file_names"]
    if dset.kind == "clean":
        df["text"] = df["super_clean"].copy()
    elif dset.kind == "other":
        df["text"] = df["kinda_clean"].copy()
    else:
        df["text"] = df["super_clean"].copy()
    df = df[
        [
            "content",
            "file",
            "test",
            "train",
            "no_frames",
            "file_names",
            "sr",
            "audio_length",
            "split",
            "text",
            "kinda_clean",
            "super_clean",
        ]
    ].reset_index(drop=True)
    return df
