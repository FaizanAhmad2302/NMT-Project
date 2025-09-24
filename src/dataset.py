import os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .vocab import CharVocab, PAD

class UrduRomanDataset(Dataset):
    def __init__(self, csv_path: str, split: str, splits_json: str = None):
        assert split in {"train", "val", "test"}
        self.df = pd.read_csv(csv_path)[["Urdu", "RomanUrdu"]]
        self.df = self.df.drop_duplicates().reset_index(drop=True)

        # Handle splits
        if splits_json is None:
            splits_json = os.path.join(os.path.dirname(csv_path), "splits.json")
        if os.path.exists(splits_json):
            with open(splits_json, "r", encoding="utf-8") as f:
                idx = json.load(f)
        else:
            n = len(self.df)
            ids = np.arange(n)
            rng = np.random.default_rng(42)
            rng.shuffle(ids)
            n_train, n_val = int(0.5*n), int(0.25*n)
            idx = {
                "train": ids[:n_train].tolist(),
                "val": ids[n_train:n_train+n_val].tolist(),
                "test": ids[n_train+n_val:].tolist()
            }
            with open(splits_json, "w", encoding="utf-8") as f:
                json.dump(idx, f)

        self.indices = idx[split]

        # Build vocab from training set only
        voc_json = os.path.join(os.path.dirname(csv_path), "vocab.json")
        if os.path.exists(voc_json):
            voc = json.load(open(voc_json, "r", encoding="utf-8"))
            self.src_vocab = CharVocab(voc["src_itos"], {ch:i for i,ch in enumerate(voc["src_itos"])})
            self.tgt_vocab = CharVocab(voc["tgt_itos"], {ch:i for i,ch in enumerate(voc["tgt_itos"])})
        else:
            train_df = self.df.iloc[idx["train"]]
            self.src_vocab = CharVocab.build_from_text(train_df["Urdu"].tolist())
            self.tgt_vocab = CharVocab.build_from_text(train_df["RomanUrdu"].tolist())
            with open(voc_json, "w", encoding="utf-8") as f:
                json.dump({
                    "src_itos": self.src_vocab.itos,
                    "tgt_itos": self.tgt_vocab.itos
                }, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.df.iloc[self.indices[i]]
        return str(row["Urdu"]), str(row["RomanUrdu"])


def collate_fn(batch, src_vocab: CharVocab, tgt_vocab: CharVocab, max_len=160):
    src_ids = [src_vocab.encode(s)[:max_len] for s, _ in batch]
    tgt_ids = [tgt_vocab.encode(t)[:max_len] for _, t in batch]

    src_max = max(len(x) for x in src_ids)
    tgt_max = max(len(x) for x in tgt_ids)

    src_pad = [x + [src_vocab.pad_id]*(src_max-len(x)) for x in src_ids]
    tgt_pad = [x + [tgt_vocab.pad_id]*(tgt_max-len(x)) for x in tgt_ids]

    return (
        torch.tensor(src_pad, dtype=torch.long),
        torch.tensor(tgt_pad, dtype=torch.long)
    )


def make_loader(csv_path, split, batch_size, shuffle):
    ds = UrduRomanDataset(csv_path, split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, ds.src_vocab, ds.tgt_vocab)
    )
    return ds, loader
