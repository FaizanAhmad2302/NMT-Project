import torch
import torch.nn as nn

from .vocab import PAD


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x = self.dropout(self.emb(src))  # [B,S,E]
        outputs, (h, c) = self.rnn(x)    # outputs: [B,S,2H]
        return outputs, (h, c)


class LuongAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.linear_in = nn.Linear(dec_hid_dim, enc_hid_dim*2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        query = self.linear_in(dec_hidden).unsqueeze(1)       # [B,1,2H]
        scores = torch.bmm(query, enc_outputs.transpose(1,2)) # [B,1,S]
        scores = scores.squeeze(1)                            # [B,S]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = self.softmax(scores)                           # [B,S]
        context = torch.bmm(attn.unsqueeze(1), enc_outputs)   # [B,1,2H]
        return context.squeeze(1), attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim,
                 num_layers=4, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.LSTM(
            emb_dim + 2*enc_hid_dim, dec_hid_dim,
            num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.attn = LuongAttention(enc_hid_dim, dec_hid_dim)
        self.fc_out = nn.Linear(dec_hid_dim + 2*enc_hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, enc_outputs):
        emb = self.dropout(self.emb(input)).unsqueeze(1)   # [B,1,E]
        context, attn = self.attn(hidden[-1], enc_outputs) # [B,2H]
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)                         # [B,H]
        logits = self.fc_out(torch.cat([output, context], dim=1))
        return logits, hidden, cell, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, enc_hid_dim, dec_hid_dim, dec_layers):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bridge_h = nn.Linear(2*enc_hid_dim, dec_layers*dec_hid_dim)
        self.bridge_c = nn.Linear(2*enc_hid_dim, dec_layers*dec_hid_dim)
        self.dec_layers = dec_layers
        self.dec_hid_dim = dec_hid_dim

    def init_decoder_state(self, h, c):
        fwd, bwd = h[-2], h[-1]                  # last fwd + bwd
        enc_last = torch.cat([fwd, bwd], dim=1)  # [B,2H]
        h0 = self.bridge_h(enc_last).view(-1, self.dec_layers, self.dec_hid_dim).transpose(0,1)
        c0 = self.bridge_c(enc_last).view(-1, self.dec_layers, self.dec_hid_dim).transpose(0,1)
        return h0.contiguous(), c0.contiguous()

    def forward(self, src, tgt, teacher_forcing=0.5):
        device = src.device
        enc_outputs, (h, c) = self.encoder(src)
        hidden, cell = self.init_decoder_state(h, c)

        B, T = tgt.size()
        outputs = []
        input_tok = tgt[:, 0]  # <sos>
        for t in range(1, T):
            logits, hidden, cell, _ = self.decoder(input_tok, hidden, cell, enc_outputs)
            outputs.append(logits.unsqueeze(1))
            pred = logits.argmax(dim=1)
            use_tf = torch.rand(1, device=device).item() < teacher_forcing
            input_tok = tgt[:, t] if use_tf else pred
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=None, sos_id=1, eos_id=2):
        # Encode
        enc_outputs, (h, c) = self.encoder(src)
        hidden, cell = self.init_decoder_state(h, c)

        B, S = src.size()
        device = src.device
        # sensible cap: source length + 10 (you can still override via arg)
        if max_len is None:
            max_len = min(160, S + 10)

        # start with <sos>
        input_tok = torch.full((B,), sos_id, dtype=torch.long, device=device)

        outputs = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, hidden, cell, _ = self.decoder(input_tok, hidden, cell, enc_outputs)
            pred = logits.argmax(dim=1)  # [B]

            outputs.append(pred.unsqueeze(1))
            # mark finished rows
            finished |= (pred == eos_id)

            # if everyone produced <eos>, we can stop early
            if finished.all():
                break

            # next input
            input_tok = pred

        # [B, L]
        return torch.cat(outputs, dim=1)

