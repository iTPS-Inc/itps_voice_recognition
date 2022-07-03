#!/usr/bin/env python3

from dsets.dset_collection.other_dset import get_other_data
from dsets.dset_collection.itps_dset import get_annotation_data
from dsets.dset_collection.librispeech_dset import get_librispeech
from dsets.dset_collection.simple_dsets import (
    get_jsut_data,
    get_nictspreds_data,
    get_ljl_data,
    get_test_data,
)
from dsets.dset_config.dset_config import DatasetConfig

import pandas as pd


datasets = {
    "librispeech": get_librispeech,
    "itps": get_annotation_data,
    "ljl": get_ljl_data,
    "jsut": get_jsut_data,
    "nict_spreds": get_nictspreds_data,
    "testing_data": get_test_data,
    "other": get_other_data,
}


def get_data(dset: DatasetConfig, **kwargs):
    return datasets[dset.name](dset, **kwargs)


def get_datasets(dset_configs, **kwargs):
    ps = []
    dfs = []
    for dset_config in dset_configs:
        p, df = get_data(dset_config, **kwargs)
        ps = ps + [p]
        df["original_dset"] = dset_config.name
        dfs = dfs + [df]
    return ps, pd.concat(dfs).reset_index(drop=True)


ENGLISH_DATASETS = [
    DatasetConfig(name="itps", lang="en"),
    DatasetConfig(name="librispeech", split="dev", kind="clean"),
    DatasetConfig(name="ljl", split="train"),
    DatasetConfig(name="nict_spreds", split="train", lang="en"),
]

JAPANESE_DATASETS = [
    DatasetConfig(name="itps", split="train", lang="jp"),
    DatasetConfig(name="jsut", split="train"),
    DatasetConfig(name="nict_spreds", split="train", lang="jp"),
    DatasetConfig(name="other", split="train", lang="jp", kind="clean"),
]

ALL_DATASETS = [
    DatasetConfig(name="itps", split="both", lang="both"),
    DatasetConfig(name="jsut", split="both"),
    DatasetConfig(name="nict_spreds", split="both", lang="all"),
    DatasetConfig(name="librispeech", split="dev", kind="clean"),
    DatasetConfig(name="ljl", split="train"),
]
