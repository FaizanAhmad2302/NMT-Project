import argparse, os, json, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.dataset import make_loader
from src.vocab import PAD
from src.model import Encoder, Decoder, Seq2Seq



def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def evaluate(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt, teacher_forcing=0.0)
            gold = tgt[:, 1:].contiguous()
            loss = crit(logits.view(-1, logits.size(-1)), gold.view(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_csv', type=str, default='data/cleaned_final.csv')
    ap.add_argument('--exp', type=str, default='base')
    ap.add_argument('--emb', type=int, default=256)
    ap.add_argument('--hid', type=int, default=512)
    ap.add_argument('--enc_layers', type=int, default=2)
    ap.add_argument('--dec_layers', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--patience', type=int, default=5,
                   help="early stopping patience (epochs)")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_ds, train_ld = make_loader(args.data_csv, 'train', args.batch, True)
    val_ds, val_ld = make_loader(args.data_csv, 'val', args.batch, False)

    src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab

    # Model
    enc = Encoder(len(src_vocab.itos), args.emb, args.hid,
                  num_layers=args.enc_layers, dropout=args.dropout)
    dec = Decoder(len(tgt_vocab.itos), args.emb, args.hid, args.hid,
                  num_layers=args.dec_layers, dropout=args.dropout)
    model = Seq2Seq(enc, dec, args.hid, args.hid, args.dec_layers).to(device)

    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    opt = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)


    # Folders
    os.makedirs('runs', exist_ok=True)
    run_dir = os.path.join('runs', args.exp)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_ld, desc=f'Epoch {epoch}/{args.epochs}')
        total = 0.0
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(src, tgt, teacher_forcing=0.5)
            gold = tgt[:, 1:].contiguous()
            loss = crit(logits.view(-1, logits.size(-1)), gold.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item()
            pbar.set_postfix(train_loss=f"{total/(pbar.n+1):.4f}")

        # Validation
        val_loss = evaluate(model, val_ld, crit, device)
        val_ppl = math.exp(min(20.0, val_loss))
        print(f"\nVal loss: {val_loss:.4f} | Val ppl: {val_ppl:.3f}")

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'src_itos': src_vocab.itos,
                'tgt_itos': tgt_vocab.itos,
                'config': vars(args)
            }, os.path.join(run_dir, 'best.pt'))
            print("✅ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("⏹ Early stopping triggered.")
                break


if __name__ == '__main__':
    main()
