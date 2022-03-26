#!/usr/bin/env python3
from pathlib import Path
from typing import Tuple

import pandas as pd
from dsets.dset_config.dset_config import DatasetConfig
from fastai.data.all import get_files, untar_data

DEV_CLEAN = "https://www.dropbox.com/s/dks1ym745vyn9l4/dev-clean.tar.gz?dl=1"
DEV_OTHER = "https://www.dropbox.com/s/hkombarzstz7mzv/dev-other.tar.gz?dl=1"
TEST_CLEAN = "https://www.dropbox.com/s/dgetfxs2pc3jeb2/test-clean.tar.gz?dl=1"
TEST_OTHER = "https://www.dropbox.com/s/efskenwqnqu68tf/test-other.tar.gz?dl=1"
TRAIN_CLEAN = "https://www.dropbox.com/s/gy1op05nv17k8ur/train-clean-360.tar.gz?dl=1"
TRAIN_OTHER = "https://www.dropbox.com/s/o1ooj5zwbgs4mwv/train-other-500.tar.gz?dl=1"

LIBRISPEECH_DSETS = {
    "dev-clean": DEV_CLEAN,
    "dev-other": DEV_OTHER,
    "test-clean": TEST_CLEAN,
    "test-other": TEST_OTHER,
    "train-clean": TRAIN_CLEAN,
    "train-other": TRAIN_OTHER,
}

lib_clean_train = DatasetConfig(name="librispeech", split="train", kind="clean")


def _get_answers_single_file(fn):
    out_dict = {}
    with open(fn, "r") as f:
        for line in f:
            line = line[:-1].split()
            filename = str(fn.parent / (line[0] + ".flac"))
            label = " ".join(line[1:])
            out_dict[filename] = label
        return out_dict


def _get_audio_files(folder):
    return get_files(folder, extensions=[".flac", ".wav"])


def _get_text_files(folder):
    return get_files(folder, extensions=[".txt"])


def _assemble_librispeech_dict(folder):
    text_files = _get_text_files(folder)
    files = {}
    for f in text_files:
        files.update(_get_answers_single_file(f))
    return files


def get_librispeech(
    dset: DatasetConfig = lib_clean_train, force_download=False
) -> Tuple[Path, dict]:
    if dset.kind is None:
        raise AttributeError(
            "Please pass one kind of ['other', 'clean', 'dev'] when requesting librispeech dataset"
        )
    dset_name = dset.split + "-" + dset.kind
    dset_url = LIBRISPEECH_DSETS[dset_name]
    path = untar_data(dset_url, force_download=force_download)

    if dset_name == "train-other":
        dset_name = "train-other-500"
    elif dset_name == "train-clean":
        dset_name = "train-clean-360"

    p, d = path, _assemble_librispeech_dict(path / dset_name)
    df = (
        pd.DataFrame(pd.Series(d))
        .reset_index()
        .rename({"index": "filename", 0: "text"}, axis="columns")
    )
    return p, df
