import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# hyperparams
batch_size = 64
block_size = 256
n_embed = 128
n_heads = 4
n_layers = 6
dropout = 0.2

max_iters = 5000
eval_interval = 300
eval_iters = 200
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"operating on device: {device}")

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# extract all unique chars from the entire corpus
all_chars = sorted(list(set(text)))
vocab_size = len(all_chars)

stoi = {char: i for i, char in enumerate(all_chars)}
itos = {i: char for i, char in enumerate(all_chars)}

encode = lambda x: [stoi[char] for char in x]
decode = lambda x: "".join([itos[i] for i in x])


# split dataset into train and validation sets
encoded_text = torch.tensor(encode(text), dtype=torch.long)
split_n_idx = int(0.9 * len(text))
train_data = encoded_text[:split_n_idx]
val_data = encoded_text[split_n_idx:]


def get_batch(split: str, batch_size: int):
    data = train_data if split.lower() == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def calculate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split, batch_size=batch_size)
            logits, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class FeedForward(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head) -> None:
        super().__init__()

        head_size = n_embed // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # get token affinites
        wei = q @ k.transpose(
            -2, -1
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei * C**-0.5  # normalize wei before softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = wei.softmax(-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embed = self.embedding_table(idx)  # (B, T, n_embed)
        pos_embed = self.pos_embedding_table(
            torch.arange(T, device=device)
        )  # (T, n_embed)
        x = token_embed + pos_embed  # (B, T, n_embed)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -block_size:]
            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
model = model.to(device)

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in tqdm(range(max_iters)):
    if step % eval_interval == 0:
        losses = calculate_loss()
        print(
            f"step {step}: train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}"
        )

    xb, yb = get_batch("train", batch_size)

    logits, loss = model(xb, targets=yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
result = model.generate(idx, max_new_tokens=500)

print(decode(result[0].tolist()))
