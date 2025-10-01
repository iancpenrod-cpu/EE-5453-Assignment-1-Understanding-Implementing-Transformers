import re
import torch
import torch.nn as nn
from collections import Counter

# -----------------------------
# Load & tokenize the corpus
# -----------------------------
SRC_PATH = "input.txt"
with open(SRC_PATH, "r", encoding="utf-8") as f:
    text = f.read()

def word_tokens(s: str):
    return re.findall(r"\b\w+\b", s.lower())

tokens = word_tokens(text)

# -----------------------------
# Build vocab (top-K)
# -----------------------------
TOP_K_VOCAB = 5000
counter = Counter(tokens)
most_common = counter.most_common(TOP_K_VOCAB)

PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for i, (w, _) in enumerate(most_common, start=2):
    vocab[w] = i
inv_vocab = {i: w for w, i in vocab.items()}
pad_id, unk_id = vocab[PAD], vocab[UNK]

def encode_toklist(toks):
    return [vocab.get(t, unk_id) for t in toks]

def decode_ids(ids):
    return " ".join(inv_vocab.get(i, "<unk>") for i in ids)

# -----------------------------
# Model (causal LM)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop1(out))
        out = self.ff(x)
        x = self.norm2(x + self.drop2(out))
        return x

class SimpleTransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=1024, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.posenc = PositionalEncoding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        x = self.posenc(x)
        causal = self._causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=causal)
        x = self.norm_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# Instantiate (untrained)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformerLanguageModel(vocab_size=len(vocab)).to(device)
model.eval()

# -----------------------------
# Generation
# -----------------------------
@torch.no_grad()
def generate_text(prompt: str, max_new_tokens=200, temperature=1.0, top_k=50):
    ids = encode_toklist(word_tokens(prompt)) or [unk_id]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]
        logits = logits / max(1e-6, temperature)
        if top_k and top_k > 0:
            vals, idxs = torch.topk(logits, k=min(top_k, logits.size(-1)))
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, idxs, vals)
            logits = mask
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    gen_ids = x[0].tolist()[len(ids):]
    return decode_ids(gen_ids)

# -----------------------------
# Generate & save
# -----------------------------
romeo_out = "ROMEO: " + generate_text("ROMEO:", max_new_tokens=200)
juliet_out = "JULIET: " + generate_text("JULIET:", max_new_tokens=200)

with open("ROMEO.txt", "w", encoding="utf-8") as f:
    f.write(romeo_out)
with open("JULIET.txt", "w", encoding="utf-8") as f:
    f.write(juliet_out)
