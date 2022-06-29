#!/usr/bin/env python3
import json
import string

import re
from fastai.data.all import TitledStr, Transform, tensor
from fastai.text.all import TensorText

EXTRA_CHARS = " .?!|-,"
HIRA = (
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだじづでどばびぶべぼぱぴぷぺぽゃゅょっぁぃぅぇぉ"
)
KATA = (
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダジヅデドバビブベボパピプペポャュョッァィゥェォ"
)


bos_regex=re.compile("^(\[UNK\])+")

class ENTransformersTokenizer(Transform):
    def __init__(self, tok=None):
        self.tokenizer = tok

    @staticmethod
    def create_vocab(outpath):
        allowed_letters = string.ascii_lowercase + EXTRA_CHARS
        print("Allowing letters: ")
        print(allowed_letters)
        vocab = {}
        special_toks = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        vocab = special_toks + [i for i in allowed_letters]
        vocab = {v: i for i, v in enumerate(vocab)}
        with open(outpath, "w") as f:
            json.dump(vocab, f)
        with open(outpath, "r") as f:
            _ = json.load(f)
        return outpath

    def encodes(self, s: str) -> TensorText:
        toks = tensor(self.tokenizer(s)["input_ids"])
        return TensorText(toks)

    def batch_decode(
        self, xs, group_tokens=True, skip_special_tokens=False, skip_unk=True, **kwargs
    ):
        if len(xs.shape) == 2:
            decoded = [
                self.tokenizer.decode(
                    x,
                    group_tokens=group_tokens,
                    skip_special_tokens=skip_special_tokens,
                    **kwargs,
                )
                for x in xs
            ]
            if skip_unk:
                decoded = [
                    bos_regex.sub(repl="", string=outstr) for outstr in decoded
                ]
            return decoded
        raise Exception("xs should be a two dimensional vector if using batch_decode")

    def decodes(
        self, x, group_tokens=True, skip_special_tokens=False, skip_unk=True, **kwargs
    ):
        outstr = self.tokenizer.decode(
            x.cpu().numpy(),
            group_tokens=group_tokens,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )
        if skip_unk:
            outstr = bos_regex.sub(repl="", string=outstr)
        return outstr


class JPTransformersTokenizer(Transform):
    trans = str.maketrans(KATA, HIRA)
    node_format_csv = r"%f[8]|"
    eos_format_csv = r"[EOS]"
    unk_format_csv = r"%m|"
    kanji = re.compile(
        "[\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+"
    )

    def __init__(self, tok=None, mcb=None, neologd_available=True, replace_unk=False):
        import MeCab

        self.neologd_available = True
        self.tokenizer = tok
        self.mcb = mcb
        if not self.mcb:
            self.mcb = MeCab.Tagger(
                f" --node-format={self.node_format_csv}"
                + f" --unk-format={self.unk_format_csv if not replace_unk else '[UNK]' }"
                + f" --eos-format={self.eos_format_csv}"
                + (
                    ""
                    if not neologd_available
                    else " -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"
                )
            )

    def kata2hira(self, s):
        return s.translate(self.trans).strip()

    def mecab_step(self, s: str):
        s = self.mcb.parse(s.lower())
        s = self.kanji.sub(self.tokenizer.unk_token, s)
        return "[BOS]|" + self.kata2hira(s)

    def encodes(self, s: str) -> TensorText:
        s = self.mecab_step(s)
        toks = tensor(self.tokenizer(s)["input_ids"])
        return TensorText(toks)

    def batch_decode(
        self, xs, group_tokens=True, skip_special_tokens=False, skip_unk=True, **kwargs
    ):
        if len(xs.shape) == 2:
            decoded = [
                self.tokenizer.decode(
                    x,
                    group_tokens=group_tokens,
                    skip_special_tokens=skip_special_tokens,
                    **kwargs,
                )
                for x in xs
            ]
            if skip_unk:
                decoded = [
                    bos_regex.sub(repl="", string=outstr)
                    for outstr in decoded
                ]
            return decoded
        raise AttributeError

    def decodes(self, x, group_tokens=False, skip_special_tokens=False,skip_unk=True, **kwargs):
        outstr = self.tokenizer.decode(
            x.cpu().numpy(),
            skip_special_tokens=skip_special_tokens,
            group_tokens=group_tokens,
            **kwargs,
        )
        if skip_unk:
            outstr = bos_regex.sub(repl="", string=outstr)
        return TitledStr(outstr)

    @staticmethod
    def create_vocab(outpath):
        extra_chars = "".join(set("、。？" + ".?!|-,\"'"))
        allowed_letters = set(
            HIRA + ".?!|-,\"'ー" + string.ascii_lowercase + extra_chars
        )
        print("Allowing letters: ")
        print(allowed_letters)
        vocab = {}
        special_toks = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        vocab = special_toks + [i for i in allowed_letters]
        vocab = {v: i for i, v in enumerate(vocab)}
        with open(outpath, "w") as f:
            json.dump(vocab, f)
        with open(outpath, "r") as f:
            _ = json.load(f)
        return outpath
