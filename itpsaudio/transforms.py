from fastai.data.all import Transform, TitledStr, tensor, ItemTransform, store_attr
from fastai.text.all import pad_chunk
import torchaudio
from itpsaudio.core import AudioTensor

class LoadAudio(Transform):
  def __init__(self, path): self.path = path
  def encodes(self, s) -> AudioTensor:
     t, _ = torchaudio.load(self.path/s)
     return t

@Transform
def capitalize(s):
    return s.upper()


class TargetProcessor(Transform):
    def __init__(self, proc):
        self.proc = proc

    def encodes(self, y):
        with self.proc.as_target_processor():
            return self.proc(y).input_ids

    def decodes(self, y):
        with self.proc.as_target_processor():
            return self.proc.decode(y)


class TransformersTokenizer(Transform):
    def __init__(self, processor):
        self.processor = processor

    def encodes(self, x):
        toks = tensor(self.processor.tokenizer(x)["input_ids"])
        return toks

    def decodes(self, x):
        return TitledStr(self.processor.tokenizer.decode(x.cpu().numpy()))


class Pad_Audio_Chunk(ItemTransform):
    "Pad `samples` by adding padding by chunks of size `seq_len`"

    def __init__(self, pad_idx=1, pad_first=True, seq_len=72, decode=True, **kwargs):
        store_attr("pad_idx, pad_first, seq_len,seq_len")
        super().__init__(**kwargs)

    def before_call(self, b):
        "Set `self.max_len` before encodes"
        self.lens_x = [x[0].shape[0] for x in b]
        self.lens_y = [x[1].shape[0] for x in b]
        self.max_len_x = max(self.lens_x)
        self.max_len_y = max(self.lens_y)

    def __call__(self, b, **kwargs):
        self.before_call(b)
        return super().__call__(tuple(b), **kwargs)

    def encodes(self, b):
        return [
            (
                pad_chunk(
                    b[i][0],
                    pad_idx=self.pad_idx,
                    pad_first=self.pad_first,
                    seq_len=self.seq_len,
                    pad_len=self.max_len_x,
                ),
                pad_chunk(
                    b[i][1],
                    pad_idx=-100,
                    pad_first=self.pad_first,
                    seq_len=self.seq_len,
                    pad_len=self.max_len_y,
                ),
            )
            for i in range(len(b))
        ]

    def decodes(self, o):
        return o[o != self.pad_idx] if self.decode else o
