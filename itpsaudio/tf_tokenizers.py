#!/usr/bin/env python3
import json
import string

from fastai.data.all import TitledStr, Transform, tensor
from fastai.text.all import TensorText

EXTRA_CHARS = " .?!|-,\"'"
HIRA = (
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだじづでどばびぶべぼぱぴぷぺぽゃゅょっぁぃぅぇぉ"
)
KATA = (
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダジヅデドバビブベボパピプペポャュョッァィゥェォ"
)


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

    def batch_decode(self, xs, group_tokens=True, skip_special_tokens=True, **kwargs):
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
            return decoded
        raise Exception("xs should be a two dimensional vector if using batch_decode")

    def decodes(self, x, group_tokens=True, skip_special_tokens=True, **kwargs):
        return TitledStr(
            self.tokenizer.decode(
                x.cpu().numpy(),
                group_tokens=group_tokens,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )
        )


class JPTransformersTokenizer(Transform):
    extra_chars = "".join(set("、。？" + ".?!|-,\"'"))

    trans = str.maketrans(KATA, HIRA)
    node_format_csv = r"%f[8]|"
    eos_format_csv = r"[EOS]\n"
    unk_format_csv = r"%m|"

    def __init__(self, tok=None, mcb=None):
        import MeCab

        self.tokenizer = tok
        self.mcb = mcb
        if not self.mcb:
            self.mcb = MeCab.Tagger(
                f" --node-format='{self.node_format_csv}'"
                + f" --unk-format='{self.unk_format_csv}'"
                + f" --eos-format='{self.eos_format_csv}'"
            )

    def kata2hira(self, s):
        return s.translate(self.trans).strip()

    def mecab_step(self, s: str):
        s = self.mcb.parse(s.lower())
        return "[BOS]" + self.kata2hira(s)

    def encodes(self, s: str) -> TensorText:
        s = self.mecab_step(s)
        toks = tensor(self.tokenizer(s)["input_ids"])
        return TensorText(toks)

    def batch_decode(self, xs, group_tokens=True, skip_special_tokens=True, **kwargs):
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
            return decoded
        raise AttributeError

    def decodes(self, x, group_tokens=False, skip_special_tokens=True, **kwargs):
        return TitledStr(
            self.tokenizer.decode(
                x.cpu().numpy(),
                skip_special_tokens=skip_special_tokens,
                group_tokens=group_tokens,
                **kwargs,
            )
        )

    @staticmethod
    def create_vocab(outpath):
        allowed_letters = set(HIRA + ".?!|-,\"'ー")
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
