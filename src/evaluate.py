import argparse, math, json
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from src.dataset import make_loader
from src.vocab import CharVocab, PAD
from src.model import Encoder, Decoder, Seq2Seq


def cer(pred: str, ref: str) -> float:
    """Character Error Rate (Levenshtein distance / len(ref))."""
    m, n = len(pred), len(ref)
    dp = np.zeros((m+1, n+1), dtype=np.int32)
    for i in range(m+1): dp[i,0] = i
    for j in range(n+1): dp[0,j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if pred[i-1] == ref[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return float(dp[m,n]) / max(1,n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_csv', type=str, default='data/cleaned_final.csv')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck['config']

    # Data
    test_ds, test_ld = make_loader(args.data_csv, 'test', args.batch, False)

    # Vocab
    src_vocab = CharVocab(ck['src_itos'], {ch:i for i,ch in enumerate(ck['src_itos'])})
    tgt_vocab = CharVocab(ck['tgt_itos'], {ch:i for i,ch in enumerate(ck['tgt_itos'])})

    # Model
    enc = Encoder(len(src_vocab.itos), cfg['emb'], cfg['hid'], cfg['enc_layers'], cfg['dropout'])
    dec = Decoder(len(tgt_vocab.itos), cfg['emb'], cfg['hid'], cfg['hid'],
                  cfg['dec_layers'], cfg['dropout'])
    model = Seq2Seq(enc, dec, cfg['hid'], cfg['hid'], cfg['dec_layers']).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()

    # Loss for perplexity
    crit = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    preds, refs = [], []
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in tqdm(test_ld, desc="Eval"):
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt, teacher_forcing=0.0)
            gold = tgt[:, 1:].contiguous()
            loss = crit(logits.view(-1, logits.size(-1)), gold.view(-1))
            total_loss += loss.item()

            gen = model.greedy_decode(
                src,
                max_len=gold.size(1) + 5,
                sos_id=tgt_vocab.sos_id,
                eos_id=tgt_vocab.eos_id
            )
            for i in range(src.size(0)):
                p = tgt_vocab.decode(gen[i].tolist(), skip_specials=True)
                r = tgt_vocab.decode(gold[i].tolist(), skip_specials=True)
                preds.append(list(p))
                refs.append([list(r)])

    ppl = math.exp(min(20.0, total_loss / max(1, len(test_ld))))
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(refs, preds, smoothing_function=smoothie)
    cer_scores = [cer("".join(p), "".join(r[0])) for p, r in zip(preds, refs)]

    print(f"Perplexity: {ppl:.3f}")
    print(f"BLEU (char-level): {bleu:.4f}")
    print(f"CER: {np.mean(cer_scores):.4f}")


if __name__ == '__main__':
    main()
