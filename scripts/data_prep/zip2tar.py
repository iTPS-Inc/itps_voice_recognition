#!/usr/bin/env python3
import sys
import os
from zipfile import ZipFile
import tarfile
import time
from pathlib import Path
from fastai.data.all import get_files, progress_bar
from local import PASSWORD

SHARE_FOLDER_ROOT = Path(Path.home() / "Dropbox/Share to iTPS AI-Team/アノテーションデータ")
SHARE_FOLDER_OUT = Path(Path.home() / "Downloads")


def zip2tar(inpf, outf):
    with ZipFile(inpf) as zipf:
        pw = None
        with tarfile.open(outf, "w:bz2") as tarf:
            for zip_info in progress_bar(zipf.infolist()):
                tar_info = tarfile.TarInfo(name=zip_info.filename)
                tar_info.size = zip_info.file_size
                try:
                    fileobj = zipf.open(zip_info.filename)
                except Exception as _:
                    fileobj = zipf.open(
                        zip_info.filename, pwd=bytes(PASSWORD, encoding="utf-8")
                    )
                tarf.addfile(tarinfo=tar_info, fileobj=fileobj)


zip_files = get_files(SHARE_FOLDER_ROOT, extensions=[".zip"])

for zfile in progress_bar(zip_files):
    outfile = SHARE_FOLDER_OUT / (Path(zfile).stem + ".tar.gz")
    print("Doing ", zfile)
    if not os.path.exists(outfile):
        zip2tar(zfile, outfile)
