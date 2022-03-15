#!/usr/bin/env python3
import os
from fastai.data.all import get_files


def _get_answers_single_file(fn):
    out_dict = {}
    with open(fn, "r") as f:
        for line in f:
            line = line[:-1].split()
            out_dict[str(fn.parent / (line[0] + ".flac"))] = " ".join(line[:1])
        return out_dict


def _get_audio_files(folder):
    return get_files(folder, extensions=[".flac", ".wav"])


def _get_text_files(folder):
    return get_files(folder, extensions=[".txt"])


def _assemple_librispeech_dict(folder):
    text_files = _get_text_files(folder)
    audio_files = _get_audio_files(folder)
    files = {}
    for f in text_files:
        files.update(_get_answers_single_file(f))
