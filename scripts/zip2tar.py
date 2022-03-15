#!/usr/bin/env python3
import sys
import os
from zipfile import ZipFile
import tarfile
import time
from pathlib import Path
from fastai.data.all import get_files,progress_bar

SHARE_FOLDER_ROOT= Path("/path/to/input/folder")
SHARE_FOLDER_OUT = Path("/path/to/output/folder")

def zip2tar(inpf, outf):
    with ZipFile(inpf) as zipf:
        with tarfile.open(outf, "w:bz2") as tarf:
            for zip_info in progress_bar(zipf.infolist()):
                tar_info = tarfile.TarInfo(name=zip_info.filename)
                tar_info.size = zip_info.file_size
                try:
                    fileobj = zipf.open(zip_info.filename)
                except RuntimeError as e:
                    pw = input(f"Please put in password for {outf}")
                    fileobj = zipf.open(zip_info.filename, pwd=pw)
                tarf.addfile(
                    tarinfo=tar_info,
                    fileobj=fileobj
                )

zip_files = get_files(SHARE_FOLDER_ROOT, extensions=[".zip"])
zip_files = list(filter(lambda x: "J-KAC" not in str(x), zip_files)) # TODO do J-KAC later (text file has zip on its own)
zip_files = list(filter(lambda x: "CENSREC-3" not in str(x), zip_files)) # TODO don't know the passwords yet
zip_files = list(filter(lambda x: "RWCP" not in str(x), zip_files)) # TODO do RWCP later  (is in two parts, make it one)

for zfile in progress_bar(zip_files):
    outfile = SHARE_FOLDER_OUT / (Path(zfile).stem + ".tar.gz")
    print("Doing ", zfile)
    if not os.path.exists(outfile):
        zip2tar(zfile, outfile)
