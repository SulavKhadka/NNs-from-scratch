import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from torch import Tensor


@dataclass
class GPTConfig:
    vocab_size: int = 128
    n_layers: int = 6
    batch_size: int = 2
    ctx_len: int = 4
    n_embed: int = 12
    n_heads: int = 4
    dropout_rate: float = 0.2
    bias: bool = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig


class CasualSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_size = config.n_embed // config.n_heads
        self.c_attn = nn.Linear(config.n_embed, 3 * self.head_size)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.bias_dropout = nn.Dropout(config.dropout_rate)
        self.resid_dropout = nn.Dropout(config.dropout_rate)
        self.register_buffer('bias', torch.tril(torch.ones((config.ctx_len, config.ctx_len))
                                                .view(1,1,config.ctx_len, config.ctx_len)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.head_size, dim=2)
        q = q.view(B, T, config.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, T, head_size)
        k = k.view(B, T, config.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, T, head_size)
        v = v.view(B, T, config.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, T, head_size)

        wei: Tensor = q @ k.transpose(-2, -1) # (B, n_heads, T, head_size) @ (B, n_heads, head_size, T) -> (B, n_heads, T, T)
        wei = wei * self.head_size ** -0.5
        wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
        wei = self.bias_dropout(wei)

        out: Tensor = wei @ v # (B, n_heads, T, T) @ (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, n_heads, T, head_size) -> (B, T, n_heads, head_size) -> (B, T, C == n_heads*head_size)
        out = self.c_proj(out)

        out = self.resid_dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embedg)
        self.attn = CasualSelfAttention()
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP()
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT2Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embed),
            'wpe': nn.Embedding(config.ctx_len, config.n_embed),
            'h': nn.ModuleList([Block() for _ in range(config.n_layers)]),
            'ln_f': nn.LayerNorm(config.n_embed)
        })
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=config.bias)
    
    def forward(self, x, targets=None):
        tok_embed = self.transformer.wte(x)
        pos_embed = self.transformer.pte(torch.arange(config.ctx_len, device=config.device))
        x = tok_embed + pos_embed

        for layer in self.transformer.h: 
            x = layer(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape

            logits = logits[B*T, C]
            targets = targets[B*T]
            loss = F.cross_entropy(logits, loss)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -config.ctx_len:]
            logits, loss = self(idx_cropped)

            logits : Tensor= logits[:, -1, :]
            probs = logits.softmax(-1)

            new_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, new_tok), dim=-1)
        return idx 
