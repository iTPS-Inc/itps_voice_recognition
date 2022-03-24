#!/usr/bin/env python3
from fastai.data.all import untar_data

TAR_FILES = [
    "1",
    "2"
]
NEW_BASE = "NEW_BASE"

file_paths = [untar_data(f) for file in TAR_FILES]
# Is this correct? two parents?
new_dir = file_paths[0].parent.parent / NEW_BASE
# Make new base directory for both zip files

os.mkdir(file_paths.parent)
for fp in file_paths:
    os.move(fp, fil)
