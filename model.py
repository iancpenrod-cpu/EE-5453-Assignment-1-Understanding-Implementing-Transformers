import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter

# ============================================================
# Step 1: Parse play into (utterance, speaker)
# ============================================================
with open("input.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

speaker_pattern = re.compile(r"^[A-Z][A-Za-z\s\']*:$")

dataset_raw = []
current_speaker = None
current_utterance = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    if speaker_pattern.match(line):
        if current_speaker and current_utterance:
            dataset_raw.append((current_speaker, " ".join(current_utterance)))
        current_speaker = line[:-1]  # drop colon
        current_utterance = []
    else:
        current_utterance.append(line)

if current_speaker and current_utterance:
    dataset_raw.append((current_speaker, " ".join(current_utterance)))

print("Sample parsed data:", dataset_raw[:5])

# ============================================================
# Step 2: Build vocabularies
# ============================================================
all_tokens = []
all_speakers = []
for speaker, utt in dataset_raw:
    tokens = re.findall(r"\b\w+\b", utt.lower())
    all_tokens.extend(tokens)
    all_speakers.append(speaker)

word_counts = Counter(all_tokens)
vocab = {"<pad>": 0, "<unk>": 1}
for i, (w, _) in enumerate(word_counts.most_common(), start=2):
    vocab[w] = i
inv_vocab = {i: w for w, i in vocab.items()}

speaker_set = sorted(set(all_speakers))
speaker2id = {spk: i for i, spk in enumerate(speaker_set)}
id2speaker = {i: spk for spk, i in speaker2id.items()}

print("Vocab size:", len(vocab))
print("Number of speakers:", len(speaker2id))

# ============================================================
# Step 3: Encode data
# ============================================================
def encode_tokens(text):
    return [vocab.get(tok, vocab["<unk>"]) for tok in re.findall(r"\b\w+\b", text.lower())]

encoded_data = [(encode_tokens(utt), speaker2id[speaker]) for speaker, utt in dataset_raw]

# ============================================================
# Step 4: PyTorch Dataset
# ============================================================
class PlayDataset(Dataset):
    def __init__(self, encoded_data, seq_len=128):
        self.seq_len = seq_len
        self.data = encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        tokens = tokens[:self.seq_len]
        if len(tokens) < self.seq_len:
            tokens = tokens + [vocab["<pad>"]] * (self.seq_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ============================================================
# Step 5: Model (Transformer for classification)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff = self.linear2(self.activation(self.linear1(x)))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        b, t = x.size()
        tok_emb = self.token_embedding(x)
        x = self.pos_encoder(tok_emb)
        causal_mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)
        x = self.ln_f(x)
        pooled = x.mean(dim=1)  # average pooling
        logits = self.classifier(pooled)
        return logits

# ============================================================
# Step 6: Train/Validation Split (90/10)
# ============================================================
dataset = PlayDataset(encoded_data, seq_len=128)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# ============================================================
# Step 7: Training loop with validation
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(
    vocab_size=len(vocab),
    num_classes=len(speaker2id),
    d_model=256,
    nhead=4,
    num_layers=4,
    dim_feedforward=1024,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

epochs = 10
for epoch in range(epochs):
    # ---- Train ----
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total
    val_loss = total_loss / len(val_loader)

    print(f"Epoch {epoch+1}: "
          f"train loss={train_loss:.4f}, train acc={train_acc:.4f}, "
          f"val loss={val_loss:.4f}, val acc={val_acc:.4f}")