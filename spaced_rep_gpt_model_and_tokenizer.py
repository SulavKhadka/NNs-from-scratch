import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

# GPT LLM
num_train_steps = 5000
learning_rate = 3e-4
batch_size = 16
eval_iter = 250

vocab_size = 512
ctx_len = 64
n_embed = 384
num_layers = 6
num_heads = 4
dropout_rate = 0.25
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
        return self.net(x) # (B, ctx_len, n_embed) -> (B, ctx_len, n_embed)

class AttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.query = nn.Linear(n_embed, head_size)
        self.key = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer('tril', torch.ones(ctx_len, ctx_len, device=device))
    
    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) # (B, ctx_len, n_embed) -> (B, ctx_len, head_size)
        k = self.key(x) # (B, ctx_len, n_embed) -> (B, ctx_len, head_size)

        wei: Tensor = q @ torch.transpose(k, -2, -1) # (B, ctx_len, head_size) @ (B, head_size, ctx_len) -> (B, ctx_len, ctx_len)
        wei = wei * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax(-1) # Normalize affinities(last dim) to add up to 1
        wei = self.dropout(wei)

        v = self.value(x) # (B, ctx_len, n_embed) -> (B, ctx_len, head_size)
        out = wei @ v # (B, ctx_len, ctx_len) @ (B, ctx_len, head_size) -> (B, ctx_len, head_size)
        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn_heads = nn.ModuleList([AttentionHead(n_embed // num_heads) for _ in range(num_heads)])
        self.attn_proj = nn.Linear(n_embed, n_embed)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward()
    
    def forward(self, x):
        mha_x = torch.cat([head(self.ln_1(x)) for head in self.attn_heads], dim=-1)
        mha_x = self.attn_proj(mha_x)
        mha_x = self.attn_dropout(mha_x)

        x = x + mha_x # (B, ctx_len, n_embed) + (B, ctx_len, n_embed)
        x = x + self.ffwd(self.ln_2(x)) # (B, ctx_len, n_embed) + (B, ctx_len, n_embed)
        return x

class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(ctx_len, n_embed)
        self.trf_block = nn.Sequential(*[Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout_rate)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x, targets = None):
        tok_embed = self.token_embedding_table(x) # (B, ctx_len) -> (B, ctx_len, n_embed)
        pos_embed = self.pos_embedding_table(torch.arange(x.shape[1], device=device)) # (B, ctx_len) -> (B, ctx_len, n_embed)
        x = tok_embed + pos_embed
        
        x = self.trf_block(x)
        x = self.ln_f(x)
        
        x = self.dropout(x)
        logits = self.lm_head(x) # (B, ctx_len, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            cropped_idx = idx[:, -ctx_len:]
            logits, loss = self(cropped_idx)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, -1) # (B, 1, vocab_size)
            new_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, new_token), dim=1)
        return idx


class BasicTokenizer():

    def __init__(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    # count of each bigram byte pair present in encoded_text
    def _get_stats(self, encoded_text):
        stats = {}
        for c1, c2 in zip(encoded_text, encoded_text[1:]):
            stats[(c1, c2)] = stats.get((c1, c2), 0) + 1
        return stats

    # replace all instances of given bigram with the specified token
    def _replace_bigram(self, encoded_text, bigram, token):
        processed_encoded_text = []
        i = 0
        while i < len(encoded_text): 
            if i < len(encoded_text) - 1 and (encoded_text[i], encoded_text[i+1]) == bigram:
                processed_encoded_text.append(token)
                i += 2
            else:
                processed_encoded_text.append(encoded_text[i])
                i += 1
        return processed_encoded_text

    def train(self, text: str, vocab_size: int):
        num_merges = vocab_size - 256
        encoded_text = list(text.encode('utf-8'))
        for i in range(num_merges):
            stats = self._get_stats(encoded_text)
            most_freq_bigram = max(stats, key=stats.get)
            self.merges[most_freq_bigram] = i + 256
            encoded_text = self._replace_bigram(encoded_text, most_freq_bigram, self.merges[most_freq_bigram])
        
        for k, v in self.merges.items():
            self.vocab[v] = self.vocab[k[0]] + self.vocab[k[1]]
    
    def encode(self, text: str):
        encoded_text = list(text.encode('utf-8'))
        while len(encoded_text) > 1:
            stats = self._get_stats(encoded_text)
            bigram_to_token_dict = {k: self.merges.get(k, float('inf')) for k, v in stats.items()}
            min_token_bigram = min(bigram_to_token_dict, key=bigram_to_token_dict.get)
            if min_token_bigram not in self.merges:
                break
            encoded_text = self._replace_bigram(encoded_text, min_token_bigram, self.merges[min_token_bigram])
        return encoded_text

    def decode(self, tokens):
        decoded_text = [self.vocab[i] for i in tokens]
        return b''.join(decoded_text).decode('utf-8')



# load text corpus and tokenize it
with open("taylorswift.txt", "r") as file:
    corpus_text = file.read()

vocab_size = 275
tokenizer = BasicTokenizer()
tokenizer.train(corpus_text, vocab_size)

tokenized_text = torch.tensor(tokenizer.encode(corpus_text))
split_idx = int(0.9 * len(tokenized_text))
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

def get_batch(batch_size: int, split: str):
    data = train_data if split == 'train' else val_data
    sample_idxs = torch.randint(len(data)-ctx_len, (batch_size,))
    x = torch.stack([data[i:i+ctx_len] for i in sample_idxs])
    y = torch.stack([data[i+1: i+1+ctx_len] for i in sample_idxs])
    x, y = x.to(device), y.to(device)
    return x, y

# training code
model = LanguageModel()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def approx_loss(num_iters: int):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.tensor(torch.zeros(num_iters))
        for i in range(num_iters):
            xb, yb = get_batch(batch_size, split)
            logits, loss = model(xb, targets=yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for i in range(num_train_steps):
    if i % eval_iter == 0:
        losses = approx_loss(eval_iter)
        print(f"step {i} | train_loss: {losses['train']} | val_loss: {losses['val']}")

    xb, yb = get_batch(batch_size, 'train')
    logits, loss = model(xb, targets=yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
losses = approx_loss(eval_iter)
print(f"Final Loss | train_loss: {losses['train']} | val_loss: {losses['val']}")


# generation
output = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=50)
print(tokenizer.decode(output[0].tolist()))