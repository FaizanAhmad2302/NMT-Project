import argparse
import torch

from src.vocab import CharVocab
from src.dataset import UrduRomanDataset
from src.model import Encoder, Decoder, Seq2Seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_csv', type=str, default='data/cleaned_final.csv')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--input_file', type=str, required=True,
                    help="Text file containing Urdu sentences (one per line)")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck['config']

    # Dataset to get vocab
    ds = UrduRomanDataset(args.data_csv, 'test')
    src_vocab = CharVocab(ck['src_itos'], {ch:i for i,ch in enumerate(ck['src_itos'])})
    tgt_vocab = CharVocab(ck['tgt_itos'], {ch:i for i,ch in enumerate(ck['tgt_itos'])})

    # Model
    enc = Encoder(len(src_vocab.itos), cfg['emb'], cfg['hid'], cfg['enc_layers'], cfg['dropout'])
    dec = Decoder(len(tgt_vocab.itos), cfg['emb'], cfg['hid'], cfg['hid'],
                  cfg['dec_layers'], cfg['dropout'])
    model = Seq2Seq(enc, dec, cfg['hid'], cfg['hid'], cfg['dec_layers']).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()

    # Read Urdu sentences from file
    with open(args.input_file, "r", encoding="utf-8") as f:
        urdu_lines = [line.strip() for line in f if line.strip()]

    for line in urdu_lines:
        ids = torch.tensor([src_vocab.encode(line)], dtype=torch.long, device=device)
        max_len = min(160, len(line) + 10)
        with torch.no_grad():
            gen = model.greedy_decode(
                ids,
                max_len=max_len,
                sos_id=tgt_vocab.sos_id,
                eos_id=tgt_vocab.eos_id
            )
        out = tgt_vocab.decode(gen[0].tolist(), skip_specials=True)
        print(f"Urdu: {line}")
        print(f"Roman Urdu: {out}")
        print("-" * 40)


if __name__ == "__main__":
    main()
