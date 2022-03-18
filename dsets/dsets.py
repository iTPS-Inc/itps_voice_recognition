#!/usr/bin/env python3
from .itps_dset import get_annotation_data
from .librispeech_dset import get_librispeech
from .simple_dsets import get_jsut_data, get_nictspreads_data, get_ljl_data
from .dset_config import DatasetConfig

import pandas as pd



datasets = {
    "librispeech": get_librispeech,
    "itps": get_annotation_data,
    "ljl": get_ljl_data,
    "jsut": get_jsut_data,
}


def get_data(dset: DatasetConfig):
    return datasets[dset.name](dset)


def get_datasets(dset_configs):
    ps = []
    dfs = []
    for dset_config in dset_configs:
        p, df = get_data(dset_config)
        ps = ps + [p]
        df["original_dset"] = dset_config.name
        dfs = dfs + [df]
    return ps, pd.concat(dfs).reset_index(drop=True)



ENGLISH_DATASETS = [
    DatasetConfig(name="itps", lang="en"),
    DatasetConfig(name="librispeech", split="dev", kind="clean"),
    DatasetConfig(name="ljl"),
]

JAPANESE_DATASETS = [
    DatasetConfig(name="itps", lang="jp"),
    DatasetConfig(name="jsut"),
]
