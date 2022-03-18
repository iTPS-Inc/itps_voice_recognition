#!/usr/bin/env python3
import requests
from dsets import get_datasets, DatasetConfig, ENGLISH_DATASETS, JAPANESE_DATASETS
import os

from dsets.librispeech import LIBRISPEECH_DSETS


def dset_ok(df):
    has_filenames = "filename" in df.columns
    has_texts = "text" in df.columns
    has_all_files = df["filename"].apply(os.path.exists).all()
    return has_filenames and has_texts and has_all_files


def test_get_ENGLISH_DATASETS():
    p, df = get_datasets(ENGLISH_DATASETS)
    assert len(df["original_dset"].unique()) == 3
    assert set(df["original_dset"].unique()) == {"itps", "librispeech", "ljl"}
    assert dset_ok(df)


def test_get_JAPANESE_DATASETS():
    p, df = get_datasets(JAPANESE_DATASETS)
    assert len(df["original_dset"].unique()) == 2
    assert set(df["original_dset"].unique()) == {"itps", "jsut"}
    assert dset_ok(df)


def test_get_itps_en():
    p, df = get_datasets(
        [DatasetConfig(name="itps", lang="en", split="both")],
    )
    assert len(p) == 1
    assert df.shape[0] == 2807
    assert dset_ok(df)


def test_get_itps_jp():
    p, df = get_datasets(
        [DatasetConfig(name="itps", lang="jp", split="both")],
    )
    assert len(p) == 1
    assert df.shape[0] == 3341
    assert dset_ok(df)


def test_get_itps_both():
    p, df = get_datasets(
        [DatasetConfig(name="itps", lang="both", split="both")],
    )
    assert len(p) == 1
    assert df.shape[0] == 6148
    assert dset_ok(df)


def test_get_itps_librispeech():
    p, df = get_datasets(
        [
            DatasetConfig(name="itps", lang="both", split="both"),
            DatasetConfig(name="librispeech", kind="other", split="dev"),
        ]
    )
    assert len(p) == 2
    assert df.shape[0] == 9012
    assert dset_ok(df)


def test_get_ljspeech():
    p, df = get_datasets([DatasetConfig(name="ljl", split="both")])
    assert df.shape == (13100, 5)
    assert dset_ok(df)


def test_get_jsut_data():
    p, df = get_datasets([DatasetConfig(name="jsut", split="both")])
    assert df.shape == (7670, 5)
    assert dset_ok(df)
