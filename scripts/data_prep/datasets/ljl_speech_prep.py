#!/usr/bin/env python3
import os
import shutil

import pandas as pd
from dsets.helpers.helpers import make_tarfile, train_test_split
from fastai.data.all import Path
from fastdownload import FastDownload
import torchaudio

LJ_SPEECH_URL_ORIG = (
    "https://www.dropbox.com/s/cwq264n040guqhj/LJSpeech-1.1.tar.bz2?dl=1"
)
DATAROOT = "/home/jjs/proj/work/itps/itps_transcription_model/scripts/data/"
OUTPATH = Path(DATAROOT) / "LJSpeech-1.1.tar.gz"
FORCE_DOWNLOAD = False


def get_ljl_data_init(base, force_download=False):
    d = FastDownload(base=base)
    p = d.get(LJ_SPEECH_URL_ORIG, force=force_download)
    df = pd.read_csv(p / "metadata.csv", sep="|", header=None)
    df[0] = df[0].apply(lambda x: p / "wavs" / (x + ".wav"))
    df = df.rename({0: "filename", 1: "text (numbers)", 2: "text"}, axis="columns")
    return p, df


p, df = get_ljl_data_init(Path(DATAROOT), force_download=FORCE_DOWNLOAD)
df = train_test_split(df)

os.mkdir(p / "wavs" / "train")
os.mkdir(p / "wavs" / "test")

for i, r in df.iterrows():
    if r["test"]:
        src = r["filename"]
        dest = r["filename"].parent / "test" / r["filename"].name
        shutil.move(src, dest)
    elif not r["test"]:
        src = r["filename"]
        dest = r["filename"].parent / "train" / r["filename"].name
        shutil.move(src, dest)


def get_frames_sr(f):
    t, sr = torchaudio.load(f)
    no_frames = len(t.squeeze())
    return pd.Series([no_frames, sr])

df[["no_frames", "sr"]] = df["filename"].apply(get_frames_sr)
df["audio_length"] = df["no_frames"] / df["sr"]

df["filename"] = df.apply(
    lambda r: Path("wavs") / "test" / r["filename"].name
    if r["test"]
    else Path("wavs") / "train" / r["filename"].name,
    axis=1,
)

df.to_csv(p / "metadata.csv")
make_tarfile(OUTPATH, p)
