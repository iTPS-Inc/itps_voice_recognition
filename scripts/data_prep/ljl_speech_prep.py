#!/usr/bin/env python3
#!/usr/bin/env python3
from fastai.data.all import untar_data
from fastai.data.transforms import RandomSplitter
import pandas as pd
import shutil, os
from .h import make_tarfile

LJ_SPEECH_URL_ORIG = "https://www.dropbox.com/s/cwq264n040guqhj/LJSpeech-1.1.tar.bz2?dl=1"
OUTPATH = "/home/jjs/Dropbox/Share to iTPS AI-Team/train_data_repository/LJSpeech-1.1.tar.gz"

def get_ljl_data_init():
    p = untar_data(LJ_SPEECH_URL_ORIG)
    df = pd.read_csv(p / "metadata.csv", sep="|", header=None)
    df[0] = df[0].apply(lambda x: p / "wavs" / (x + ".wav"))
    df = df.rename({0: "filename", 1: "text (numbers)", 2: "text"}, axis="columns")
    return p, df


p, df = get_ljl_data_init()
splits = RandomSplitter(seed=42, valid_pct=0.2)(df)
df["test"] = False
df.loc[splits[1], "test"] = True

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

df.apply(print, axis=1)

df["filename"] = df.apply(
    lambda r: r["filename"].parent / "test" / r["filename"].name
    if r["test"]
    else r["filename"].parent / "train" / r["filename"].name
    ,axis=1
)
df.to_csv( p / "metadata.csv")

make_tarfile(OUTPATH, p)
