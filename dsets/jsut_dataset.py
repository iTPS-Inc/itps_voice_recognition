#!/usr/bin/env python3
from fastai.data.all import untar_data, get_files
import pandas as pd

JSUT_URL = "https://www.dropbox.com/s/a2z6bklpaphiu1k/jsut_ver1.1.zip?dl=1"


def _get_info_file(fn):
    out_dict = {}
    with open(fn, "r") as f:
        for line in f:
            line = line.split(":")
            filename = str(fn.parent / "wav" / (line[0] + ".wav"))
            label = ":".join(line[1:]).strip()
            out_dict[filename] = label
        return out_dict


def _get_info(text_files, kind="transcript_utf8", colname="text"):
    d = {}
    for f in text_files:
        if f.stem == kind:
            d.update(_get_info_file(f))
    return (
        pd.DataFrame(pd.Series(d))
        .reset_index()
        .rename({"index": "filename", 0: colname}, axis="columns")
    )


def get_jsut_data():
    p = untar_data(JSUT_URL)
    text_files = get_files(p, extensions=[".txt"])
    d = _get_info(text_files, kind="transcript_utf8", colname="text")
    df = _get_info(text_files, kind="recording_info", colname="recording_info")
    d = pd.merge(d, df, on="filename", how="inner")
    return p, d
