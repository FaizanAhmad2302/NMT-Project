from dataclasses import dataclass
from typing import List, Dict

# Special tokens
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD, SOS, EOS, UNK = range(4)

@dataclass
class CharVocab:
    itos: List[str]       # index to string
    stoi: Dict[str, int]  # string to index

    @classmethod
    def build_from_text(cls, texts: List[str]):
        charset = set()
        for t in texts:
            charset.update(list(t))
        # specials + sorted chars
        base = SPECIALS + sorted(list(charset))
        stoi = {ch: i for i, ch in enumerate(base)}
        return cls(base, stoi)

    def encode(self, s: str, add_sos_eos=True) -> List[int]:
        ids = [self.stoi.get(ch, UNK) for ch in s]
        if add_sos_eos:
            return [SOS] + ids + [EOS]
        return ids

    def decode(self, ids: List[int], skip_specials=True) -> str:
        out = []
        for i in ids:
            if skip_specials and i in (PAD, SOS, EOS):
                continue
            out.append(self.itos[i])
        return "".join(out)

    @property
    def pad_id(self):
        return PAD

    @property
    def sos_id(self):
        return SOS

    @property
    def eos_id(self):
        return EOS
