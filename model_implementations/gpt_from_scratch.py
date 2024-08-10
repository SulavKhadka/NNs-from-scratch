import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math

class GPTConfig:
    vocab_size: int = 256
    batch_size: int = 8
    ctx_len: int = 16
    n_layers: int = 6
    n_heads: int = 4
    n_embed: int = 32
    dropout_rate: float = 0.2


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.in_proj = nn.Linear(config.n_embed, config.n_embed * 4)
        self.out_proj = nn.Linear(config.n_embed * 4, config.n_embed)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.in_proj(x) # (B, T, C) -> (B, T, C * 4)
        x = F.relu(x) # (B, T, C * 4)
        x = self.out_proj(x) #(B, T, C * 4) -> (B, T, C)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.head_size = config.n_embed // config.n_heads

        self.q = nn.Linear(config.n_embed, self.head_size)
        self.k = nn.Linear(config.n_embed, self.head_size)
        self.v = nn.Linear(config.n_embed, self.head_size)
        self.register_buffer("tril", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
    
    def forward(self, x):
        B, T, C = x.shape

        query = self.q(x) # (B, T, C) -> (B, T, head_size)
        key = self.k(x)
        value = self.v(x)

        # wei: Tensor = (query @ key.transpose()) / math.sqrt(self.head_size) # non einsum version
        wei = torch.einsum('ijk, ilk -> ijl', [query, key]) / math.sqrt(self.head_size) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # out = wei @ value # non einsum version
        out = torch.einsum('ijk, ikl -> ijl', [wei, value]) # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.heads = [SelfAttention(config) for _ in config.n_heads]
        self.proj = nn.Linear(config.n_embed, config.n_embed)
    
    def forward(self, x):
        indiv_head_outputs = [head(x) for head in self.heads] # [(B, T, head_size), (B, T, head_size), ...]
        out = torch.cat(indiv_head_outputs, dim=-1) # (B, T, C)
        out = self.proj(out)
        return out


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_pre_attn = nn.LayerNorm(config.n_embed)
        self.mha_attn = MultiHeadAttention(config)
        self.ln_pre_mlp = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mha_attn(self.ln_pre_attn(x))
        x = x + self.mlp(self.ln_pre_mlp(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.tok_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = nn.Embedding(config.ctx_len, config.n_embed)
        self.blocks = [GPTBlock(config) for _ in config.n_layers]
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, x, targets = None):
        B, T = x.shape

        embed_tok = self.tok_embed(x) #(B, T, C)
        embed_pos = self.pos_embed(torch.arange(T, device=x.device()))
        x = embed_tok + embed_pos

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, x, max_new_tokens: int):
        B, T = x.shape

        for _ in max_new_tokens:
            x_cropped = x[:, -T:]
            logits, loss = self(x_cropped)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            new_idx = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, new_idx], dim=1)
        
        return x




