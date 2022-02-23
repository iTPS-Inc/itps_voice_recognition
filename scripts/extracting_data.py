#!/usr/bin/env python3
from pathlib import Path
import os
import json
import pandas as pd
import shutil

datapath = Path("/Users/jiyanschneider/proj/work/itps/itps_transcription_model/data/")
root_foldername = "annotation_data/"
if not root_foldername.endswith("/"):
    root_foldername += "/"

raw_path = (Path(datapath / "raw" / root_foldername)).resolve()

processed_path = (Path(datapath) / "processed" / "annotation_data").resolve()

excel_files = []
colmap = {
    "Source Video Name": "source_video",
    "Source Audio Name": "source_video",
    "Segmented audio file name": "file",
    "Transcription": "text",
    "Keywords": "keywords",
    "Comment": "comment",
    "st": "start_time",
    "et": "end_time",
}

othermap = {
    "Column4": "source_video",
    "Column5": "file",
    "Column6": "text",
    "Column7": "keywords",
    "Column8": "comment",
}


def rename_columns(df):
    if "Column4" not in df.columns:
        df = df.rename(colmap, axis=1)
    else:
        df = df.rename(
            othermap,
            axis=1,
        )
        if df["source_video"].iloc[0] == "input1":
            df = df.loc[1:, :]
        df = df[list(othermap.values())]
    return df


def process_csv(filepath):
    df = pd.read_csv(filepath)
    sname = subfoldername(filepath)
    parent = str(Path(filepath).parent)
    df = rename_columns(df)
    df["filepath"] = filepath
    df[["st", "text", "et"]] = df["text"].str.extract(r"(\[.*\])(.*)(\[.*\])")
    df["old_path"] = str(parent) + "/" + df["file"]
    df["new_filename"] = df["file"].str.extract(r"_(\d+)\.wav$") + ".wav"
    df["new_rel_path"] = sname + df["new_filename"]
    df["audio_filename"] = df["new_rel_path"]
    return df


x = 0


def move_files(test, verbose=False):
    tdf = df[df[test]].copy()
    for _, row in tdf.iterrows():
        old_fp: str = row["old_path"]  # type: ignore
        datadir = Path(old_fp.split("raw")[0])
        processed = processed_path / test
        new_relp = row["audio_filename"]
        new_fp = Path(str(processed) + "/" + str(new_relp))
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(old_fp)
            print(new_fp)
        try:
            shutil.copyfile(old_fp, new_fp)
        except Exception as e:
            print(e)
            print(old_fp, new_fp, row)
            global x
            x = row
            break


def subfoldername(f):
    subfn = f.split(root_foldername)[1]
    dirnames = subfn.split("/")
    if "Japanese" in dirnames[0]:
        dirnames[0] = "jp"
    elif "English" in dirnames[0]:
        dirnames[0] = "en"
    if "." in dirnames[1]:
        dirnames[1] = dirnames[1].split(".")[0]
    dirnames = dirnames[:-1]
    return "/".join(dirnames) + "/"


excel_files = []
for dp, dn, fn in os.walk(raw_path):
    for f in fn:
        if f.endswith(".csv"):
            df = process_csv(f"{dp}/{f}")
            excel_files.append(df)

df = pd.concat(excel_files, ignore_index=True)
df = df.dropna(how="all")

randperm = df.sample(frac=1, random_state=123).index
idx = int(df.shape[0] * 0.8)

df.loc[randperm[:idx], "train"] = True
df["train"] = df["train"].fillna(False)

df["test"] = ~df["train"]
df["test"] = df["test"].fillna(False)

train = df[df["train"]].copy()
test = df[df["test"]].copy()

move_files("train")
move_files("test")

df = df.drop(["new_rel_path", "old_path"], axis=1)

randperm = df.sample(frac=1, random_state=123).index
idx = int(df.shape[0] * 0.8)

df.loc[randperm[:idx], "train"] = True
df["train"] = df["train"].fillna(False)

df["test"] = ~df["train"]
df["test"] = df["test"].fillna(False)


train = df[df["train"]].copy()
test = df[df["test"]].copy()
train = train.drop(["train", "test"], axis=1)
test = test.drop(["train", "test"], axis=1)

full = df.copy()
train.to_json(processed_path / "train.json", orient="records")
test.to_json(processed_path / "test.json", orient="records")
full.to_json(processed_path / "full.json", orient="records")

train.to_json(processed_path / "train.json", orient="records")


test.to_json(processed_path / "test.json", orient="records")
full.to_json(processed_path / "full.json", orient="records")
