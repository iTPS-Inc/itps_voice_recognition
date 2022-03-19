#!/usr/bin/env python3
import os, shutil
from pathlib import Path

import pandas as pd
from fastai.data.all import get_files, untar_data
from tqdm import tqdm

from dsets.helpers.helpers import make_tarfile, train_test_split

JSUT_URL_ORIG = "https://www.dropbox.com/s/a2z6bklpaphiu1k/jsut_ver1.1.zip?dl=1"
OUTPATH = "/home/jjs/Dropbox/Share to iTPS AI-Team/train_data_repository/jsut_ver1.1.tar.gz"
FORCE_DOWNLOAD=False

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


def get_jsut_data(force_download=True):
    p = untar_data(JSUT_URL_ORIG, force_download=force_download)
    text_files = get_files(p, extensions=[".txt"])
    d = _get_info(text_files, kind="transcript_utf8", colname="text")
    df = _get_info(text_files, kind="recording_info", colname="recording_info")
    d = pd.merge(d, df, on="filename", how="inner")
    return p, d


p, df = get_jsut_data(force_download=FORCE_DOWNLOAD)
df = train_test_split(df)

for i in p.ls():
    if not os.path.isdir(i): continue
    if not os.path.exists(i / "wav"/ "train"): os.mkdir( i / "wav"/ "train" )
    if not os.path.exists(i / "wav"/ "test"): os.mkdir( i / "wav"/ "test" )

for i, r in tqdm(df.iterrows()):
    if r["test"]:
        src = r["filename"]
        dest = r["filename"].parent / "test" / r["filename"].name
        if os.path.exists(src) and not os.path.exists(dest):
            shutil.move(src, dest)
    elif not r["test"]:
        src = r["filename"]
        dest = r["filename"].parent / "train" / r["filename"].name
        if os.path.exists(src) and not os.path.exists(dest):
            shutil.move(src, dest)


df["filename"] = df.apply(
    lambda r: r["filename"].parent / "test" / r["filename"].name
    if r["test"]
    else r["filename"].parent / "train" / r["filename"].name,
    axis=1,
)
assert df["filename"].apply(os.path.exists).all()
df.to_csv(p / "metadata.csv")
make_tarfile(OUTPATH, p)
