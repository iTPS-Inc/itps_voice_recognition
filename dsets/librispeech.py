#!/usr/bin/env python3
import os
from fastai.data.all import get_files, untar_data
from pathlib import Path
from typing import Literal, Tuple
import pandas as pd
import requests

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


Dsets = Literal["clean", "other"]
Dset_types = Literal["train", "test", "dev"]

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

def get_librispeech(dset: Dsets, dset_type: Dset_types) -> Tuple[Path, dict]:
    dset_name = dset + "-" + dset_type
    dset_url = LIBRISPEECH_DSETS[dset_name]
    path = untar_data(dset_url)
    p, d = path, _assemble_librispeech_dict(path / dset_name)
    df = (
        pd.DataFrame(pd.Series(d))
        .reset_index()
        .rename({"index": "filename", 0: "text"}, axis="columns")
    )
    return p, df
