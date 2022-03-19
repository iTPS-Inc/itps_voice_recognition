#!/usr/bin/env python3
import os

from dsets.dsets import (ALL_DATASETS, ENGLISH_DATASETS, JAPANESE_DATASETS,
                         DatasetConfig, get_datasets)
from dsets.helpers.helpers import get_sampling_rates

HAVE_TIME = False
REDOWNLOAD = False
NUM_CORES = 8


def dset_ok(df):
    has_filenames = "filename" in df.columns
    has_texts = "text" in df.columns
    has_all_files = df["filename"].apply(os.path.exists).all()
    if HAVE_TIME:
        sr = get_sampling_rates(df["filename"], NUM_CORES)
        assert (~(sr.isna())).sum() == df.shape[0]
    return has_filenames and has_texts and has_all_files


def test_get_ENGLISH_DATASETS():
    p, df = get_datasets(ENGLISH_DATASETS, force_download=REDOWNLOAD)
    assert len(df["original_dset"].unique()) == 4
    assert set(df["original_dset"].unique()) == {
        "itps",
        "librispeech",
        "ljl",
        "nict_spreds",
    }
    assert dset_ok(df)


def test_get_JAPANESE_DATASETS():
    p, df = get_datasets(JAPANESE_DATASETS, force_download=REDOWNLOAD)
    assert len(df["original_dset"].unique()) == 3
    assert set(df["original_dset"].unique()) == {"itps", "jsut", "nict_spreds"}
    assert dset_ok(df)


def test_get_itps_en():
    p, df = get_datasets(
        [DatasetConfig(name="itps", lang="en", split="both")], force_download=REDOWNLOAD
    )
    assert len(p) == 1
    assert df.shape[0] >= 2799
    assert dset_ok(df)


def test_get_itps_jp():
    p, df = get_datasets(
        [DatasetConfig(name="itps", lang="jp", split="both")], force_download=REDOWNLOAD
    )
    assert len(p) == 1
    assert df.shape[0] >= 3329
    assert dset_ok(df)


def test_get_itps_both():
    p, df = get_datasets(
        [DatasetConfig(name="itps", lang="both", split="both")],
        force_download=REDOWNLOAD,
    )
    assert len(p) == 1
    assert df.shape[0] >= 6128
    assert dset_ok(df)


def test_get_itps_librispeech():
    p, df = get_datasets(
        [
            DatasetConfig(name="itps", lang="both", split="both"),
            DatasetConfig(name="librispeech", kind="other", split="dev"),
        ],
        force_download=REDOWNLOAD,
    )
    assert len(p) == 2
    assert df.shape[0] >= 8992
    assert dset_ok(df)


def test_get_ljspeech():
    p, df = get_datasets(
        [DatasetConfig(name="ljl", split="both")], force_download=REDOWNLOAD
    )
    assert df.shape == (13100, 5)
    assert dset_ok(df)


def test_get_jsut_data():
    p, df = get_datasets(
        [DatasetConfig(name="jsut", split="both")], force_download=REDOWNLOAD
    )
    assert df.shape == (7670, 5)
    assert dset_ok(df)


def test_get_nict_data():
    p, df = get_datasets(
        [DatasetConfig(name="nict_spreds", split="both", lang="all")],
        force_download=REDOWNLOAD,
    )
    assert not "ja" in set(df.language.unique())
    assert "jp" in set(df.language.unique())
    assert df.shape == (12000, 15)
    assert dset_ok(df)


def test_get_nict_data_ja():
    p, df = get_datasets(
        [DatasetConfig(name="nict_spreds", split="both", lang="jp")],
        force_download=REDOWNLOAD,
    )
    assert df.shape == (1000, 15)
    assert dset_ok(df)
    assert not "ja" in set(df.language.unique())
    assert "jp" in set(df.language.unique())
    p, df = get_datasets(
        [DatasetConfig(name="nict_spreds", split="train", lang="jp")],
        force_download=REDOWNLOAD,
    )
    assert df.shape == (800, 15)
    assert dset_ok(df)


def test_dataloading():
    p, df = get_datasets(ALL_DATASETS, force_download=REDOWNLOAD)
    assert df.shape[0] >= 30000
    assert dset_ok(df)
