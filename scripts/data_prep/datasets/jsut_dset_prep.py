#!/usr/bin/env python3
import os, shutil
from pathlib import Path

import torchaudio
import pandas as pd
from fastai.data.all import get_files
from tqdm import tqdm
from fastdownload import FastDownload

from dsets.helpers.helpers import make_tarfile, train_test_split

JSUT_URL_ORIG = "https://www.dropbox.com/s/a2z6bklpaphiu1k/jsut_ver1.1.zip?dl=1"
DATAROOT = os.environ.get("PREPROCESS_DATAROOT", str(Path.home() / ".fastdownload"))
OUTPATH = Path(DATAROOT)/ "out" / "jsut_ver1.1.tar.gz"
FORCE_DOWNLOAD = True


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


def get_jsut_data(base, force_download=True):
    d = FastDownload()
    p = d.get(JSUT_URL_ORIG, force=force_download)
    text_files = get_files(p, extensions=[".txt"])
    d = _get_info(text_files, kind="transcript_utf8", colname="text")
    df = _get_info(text_files, kind="recording_info", colname="recording_info")
    d = pd.merge(d, df, on="filename", how="inner")
    return p, d


p, df = get_jsut_data(str(DATAROOT), force_download=FORCE_DOWNLOAD)
df = train_test_split(df)

for i in p.ls():
    if not os.path.isdir(i):
        continue
    if not os.path.exists(i / "wav" / "train"):
        os.mkdir(i / "wav" / "train")
    if not os.path.exists(i / "wav" / "test"):
        os.mkdir(i / "wav" / "test")

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


def get_frames_sr(f):
    t, sr = torchaudio.load(f)
    no_frames = len(t.squeeze())
    return pd.Series([no_frames, sr])


df[["no_frames", "sr"]] = df["filename"].apply(get_frames_sr)
df["audio_length"] = df["no_frames"] / df["sr"]

df["filename"] = df.apply(
    lambda r: Path(".")
    / r["filename"].parent.parent.parent.name
    / r["filename"].parent.parent.name
    / "test"
    / r["filename"].name
    if r["test"]
    else Path(".")
    / r["filename"].parent.parent.parent.name
    / r["filename"].parent.parent.name
    / "train"
    / r["filename"].name,
    axis=1,
)

df.to_csv(p / "metadata.csv")
make_tarfile(OUTPATH, p)
