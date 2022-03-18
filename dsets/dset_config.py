#!/usr/bin/env python3
from dataclasses import dataclass
from enum import Enum
from typing import Optional

Name = Enum("name", ["itps", "librispeech", "jsut", "ljl", "nict_spreds"])
Split = Enum("Split", ["train", "test", "both", "dev"])
Language = Enum(
    "language", ["jp", "en", "ru", "zh", "th", "es", "br", "fr", "vi", "ko", "id", "my"]
)
Kind = Enum("Kind", ["other", "clean"])


@dataclass
class DatasetConfig:
    name: Name
    split: Split = "train"
    lang: Optional[Language] = None
    kind: Optional[Kind] = None
