import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from dataclasses import dataclass
import math

@dataclass
class LlamaConfig:
    vocab: int = 32_000
    n_embd: int = 4096
    intermediate_size: int = 11008
    
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 16

    batch_size: int = 32
    ctx_len: int = 4096

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('tril', torch.tril(torch.ones((config.ctx_len, config.ctx_len), dtype=torch.long)).view(1, 1, 1, config.ctx_len, config.ctx_len))

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embd

    def forward(self, x):
        B, T, C = x.shape

        query = self.q_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, C // n_heads)
        key = self.k_proj(x).view(B, T, self.n_kv_heads, C // self.n_kv_heads).transpose(1, 2) # (B, kv_heads, T, C // kv_heads)
        value = self.v_proj(x).view(B, T, self.n_kv_heads, C // self.n_kv_heads).transpose(1, 2) # (B, kv_heads, T, C // kv_heads)

        query = query.view(B, self.n_heads // self.n_kv_heads, self.n_kv_heads, T, C // self.n_heads)
        key = key.unsqueeze(1).expand(-1, self.n_heads // self.n_kv_heads, -1, -1, -1)
        value = value.unsqueeze(1).expand(-1, self.n_heads // self.n_kv_heads, -1, -1 ,1)

        wei: Tensor = (query @ key.transpose(-2, -1)) / math.sqrt(self.n_heads) # (B, n_heads // kv_heads, kv_heads, T, C // n_heads) @ (B, n_heads // kv_heads, kv_heads, T, C // kv_heads) -> (B, n_heads // kv_heads, kv_heads, T, T)
        wei = wei.masked_fill(self.tril[:, :, :, :T, :T] == 0, float('-inf'))
        wei = wei.softmax(-1)

        out = wei @ value # (B, n_heads // kv_heads, kv_heads, T, T) @ (B, n_heads // kv_heads, kv_heads, T, C // kv_heads) -> (B, n_heads // kv_heads, kv_heads, T, C // kv_heads)

class MLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__() 

        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd)

    def forward(self, x):
        x = F.silu(self.up_proj(x)) * self.gate_proj(x)
        x = self.down_proj(x)
        # TODO: another dropout here?
        return x

class Block(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.self_attn = GroupedQueryAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.n_embd)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        # TODO: probably a dropout here
        return x

class Llama2Model(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.RMSNorm(config.n_embd)

    def forward(self, x):
        # x.shape = (B, T)
        x = self.embed_tokens(x) # (B, T, C)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)

        return x
