import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from tqdm import tqdm

with open("mishmash_input.txt", "r") as f:
    corpus_text = f.read()

# char tokenizer
# vocab = sorted(list(set(corpus_text)))
# vocab_size = len(vocab)

# print(vocab)

# stoi = {c:i for i, c in enumerate(vocab)}
# itos = {i: c for i, c in enumerate(vocab)}

# encode = lambda x: [stoi[i] for i in x]
# decode = lambda x: "".join([itos[i] for i in x])

# BPE tokenizer
from spaced_rep_gpt_tokenizer import BasicTokenizer, RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train(corpus_text, 384)

encode = lambda x: tokenizer.encode(x)
decode = lambda x: tokenizer.decode(x)

vocab = tokenizer.vocab
vocab_size = len(vocab)

with open("notebooks/output_text.txt", "r") as f:
    corpus_text = f.read()

batch_size = 128
ctx_len = 256
n_embed = 256
num_heads = 4
num_layers = 6
dropout_rate = 0.25

train_steps = 25_000
eval_interval = 500
eval_iters = 250
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.net(x)

class AttentionHead(nn.Module):

    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer('tril', torch.tril(torch.ones(ctx_len, ctx_len)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) # (B, ctx_len, n_embed) -> (B, ctx_len, head_size)
        k = self.key(x) # (B, ctx_len, n_embed) -> (B, ctx_len, head_size)

        k_t = k.transpose(-2, -1) # (B, ctx_len, head_size) -> (B, head_size, ctx_len)
        wei: Tensor = q @ k_t # (B, ctx_len, head_size) @ (B, head_size, ctx_len) -> (B, ctx_len, ctx_len)
        wei = wei * k.shape[-1] ** -0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax(-1) # (B, ctx_len, ctx_len)
        wei = self.dropout(wei)

        v = self.value(x) # (B, ctx_len, n_embed) -> (B, ctx_len, head_size)
        out = wei @ v # (B, ctx_len, ctx_len) @ (B, ctx_len, head_size) -> (B, ctx_len, head_size)

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.head_size = n_embed // num_heads
        assert self.head_size * num_heads == n_embed
        self.sa_heads = nn.ModuleList([AttentionHead(self.head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.sa_heads], dim=-1) # concat([(B, ctx_len, head_size) for head in sa_heads ]) -> (B, ctx_len, head_size * len(sa_heads))
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embed)
        self.mha = MultiHeadAttention()
        self.ln_2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

class GPTLanguageModel(nn.Module):

    # we need to init the current class and its parent nn.Module
    def __init__(self, ) -> None:
        super().__init__()

        self.tok_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(ctx_len, n_embed)
        self.sa_blocks = nn.Sequential(*[Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        token_embed = self.tok_embedding_table(x) # (B, ctx_len, 1) -> (B, ctx_len, n_embed)
        pos_embed = self.pos_embedding_table(torch.arange(T, device=device)) # (ctx_len, 1) -> (ctx_len, n_embed)

        x = token_embed + pos_embed # (B, ctx_len, n_embed) + (, ctx_len, n_embed) -> (B, ctx_len, n_embed)
        
        x = self.sa_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, ctx_len, n_embed)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in tqdm(range(max_new_tokens)):
            x_cropped = x[:, -ctx_len:] # (B, ctx_len)
            logits, loss = self(x_cropped) # logits: (B, ctx_len, n_embed)
            logits = logits[:, -1, :] # (B, 1, n_embed)
            
            probs = F.softmax(logits, -1)
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1, n_embed)
            x = torch.cat((x, x_next), dim=1) # (B, ctx_len, n_embed) + (B, 1, n_embed)
        return x
    


model = GPTLanguageModel()
model = model.to(device)

# data load and train/test split creation
encoded_corpus = torch.tensor(encode(corpus_text), dtype=torch.long)
split_idx = int(0.9 * len(encoded_corpus))
print(split_idx)
train_data = encoded_corpus[:split_idx]
val_data = encoded_corpus[split_idx:]

# batching function
def get_batch(split: str, batch_size: int):
    data = train_data if split == 'train' else val_data
    sample_idxs = torch.randint(len(data)-ctx_len, (batch_size,))
    x = torch.stack([data[i:i+ctx_len] for i in sample_idxs])
    y = torch.stack([data[i+1:i+1+ctx_len] for i in sample_idxs])
    x, y = x.to(device), y.to(device)
    return x, y

# estimate loss
@torch.no_grad()
def approximate_loss(num_iters: int):
    model.eval()
    out = {}
    # need both train and val losses
    for split in ['train', 'val']:
        losses = torch.zeros(num_iters)
        for i in range(num_iters):
            xb, yb = get_batch(split, batch_size)
            logits, loss = model(xb, targets=yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


# training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"training on {device}")

from torch.utils.tensorboard import SummaryWriter

# Add TensorBoard SummaryWriter
writer = SummaryWriter()

# Edit the training loop to log losses to TensorBoard
for i in tqdm(range(train_steps)):
    if i % eval_interval == 0:
        losses = approximate_loss(eval_iters)
        print(f"step: {i} | train loss: {losses['train']} | val loss: {losses['val']}")
        
        # Log losses to TensorBoard
        writer.add_scalar('Loss/train', losses['train'], i)
        writer.add_scalar('Loss/val', losses['val'], i)

    xb, yb = get_batch('train', batch_size)
    logits, loss = model(xb, targets=yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Close the TensorBoard writer after training is finished
writer.close()

# generation
x, y = get_batch('train', 1)
output = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)
print(decode(output[0].tolist()))