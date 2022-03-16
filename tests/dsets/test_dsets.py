#!/usr/bin/env python3
import requests
from dsets import get_datasets, DatasetConfig, ENGLISH_DATASETS, JAPANESE_DATASETS
import os

from dsets.librispeech import LIBRISPEECH_DSETS

def test_get_itps_librispeech():
    p, df = get_datasets(
        [
            DatasetConfig(name="itps", lang="both", split="both"),
            DatasetConfig(name="librispeech", kind="other", split="dev"),
        ]
    )
    assert len(p) == 2
    assert df.shape[0] == 9012
    assert df["filename"].apply(os.path.exists).all()

def test_get_ENGLISH_DATASETS():
    p, df = get_datasets(ENGLISH_DATASETS)
    assert df["filename"].apply(os.path.exists).all()
    assert len(df["original_dset"].unique()) == 3
    assert set(df["original_dset"].unique()) == {"itps", "librispeech", "ljl"}

def test_get_JAPANESE_DATASETS():
    p, df = get_datasets(JAPANESE_DATASETS)
    assert df["filename"].apply(os.path.exists).all()
    assert len(df["original_dset"].unique()) == 2
    assert set(df["original_dset"].unique()) == {"itps", "jsut"}
