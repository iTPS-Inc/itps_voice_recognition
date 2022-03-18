#!/usr/bin/env python3
from dsets import get_datasets, DatasetConfig, ENGLISH_DATASETS, JAPANESE_DATASETS
import os

def dset_ok(df):
    has_filenames = "filename" in df.columns
    has_texts = "text" in df.columns
    has_all_files = df["filename"].apply(os.path.exists).all()
    return has_filenames and has_texts and has_all_files


def test_get_ENGLISH_DATASETS():
    p, df = get_datasets(ENGLISH_DATASETS)
    assert len(df["original_dset"].unique()) == 4
    assert set(df["original_dset"].unique()) == {"itps", "librispeech", "ljl", "nict_spreds"}
    assert dset_ok(df)


def test_get_JAPANESE_DATASETS():
    p, df = get_datasets(JAPANESE_DATASETS)
    assert len(df["original_dset"].unique()) == 3
    assert set(df["original_dset"].unique()) == {"itps", "jsut", "nict_spreds"}
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


def test_get_nict_data():
    p, df = get_datasets([DatasetConfig(name="nict_spreds", split="both", lang="all")])
    assert not "ja" in set(df.language.unique())
    assert "jp" in set(df.language.unique())
    assert df.shape == (12000,15)
    assert dset_ok(df)


def test_get_nict_data_ja():
    p, df = get_datasets([DatasetConfig(name="nict_spreds", split="both", lang="jp")])
    assert df.shape == (1000,15)
    assert dset_ok(df)
    assert not "ja" in set(df.language.unique())
    assert "jp" in set(df.language.unique())
    p, df= get_datasets([DatasetConfig(name="nict_spreds", split="train", lang="jp")])
    assert df.shape == (800,15)
    assert dset_ok(df)
