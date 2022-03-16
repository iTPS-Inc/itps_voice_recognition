#!/usr/bin/env python3
from .itps_corpus import get_annotation_data
from .librispeech import get_librispeech
from .jsut_dataset import get_jsut_data
from .ljl_dataset import get_ljl_data
from .dset_config import DatasetConfig, Name

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
