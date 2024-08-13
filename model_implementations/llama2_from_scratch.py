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

    device: str = 'cuda'

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.head_dim = config.n_embd // config.n_heads

        # compute the theta_freqs since they are constant vectors applied to any given input x.
        theta_numerator = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        thetas = 1 / torch.pow(10_000, theta_numerator / self.head_dim).to(config.device)
        m = torch.arange(0, config.ctx_len).to(config.device)
        
        freqs = torch.einsum('i, j -> ij', [m, thetas]).float()
        self.freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer('freqs_complex', self.freqs_complex)

    def forward(self, x):
        # x.shape = (B, ctx_len, n_heads, head_dim) where n_heads can be either [n_heads, n_kv_heads]
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x) # (B, ctx_len, n_heads, head_dim // 2, 2) -> (B, ctx_len, n_heads, head_dim // 2)

        self.freqs_complex = self.freqs_complex.unsqueeze(0).unsqueeze(2) # (ctx_len, head_dim // 2) -> (1, ctx_len, head_dim // 2) -> (1, ctx_len, 1, head_dim // 2)
        x_rotated = x_complex * self.freqs_complex

        x_out = torch.view_as_real(x_rotated) #(B, ctx_len, n_heads, head_dim // 2) -> (B, ctx_len, n_heads, head_dim // 2, 2)
        x_out = x_out.reshape(*x.shape) # (B, ctx_len, n_heads, head_dim // 2, 2) -> (B, ctx_len, n_heads, head_dim)
        return x_out

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.head_dim = config.n_embd // config.n_heads

        self.q_proj = nn.Linear(config.n_embd, config.n_heads * self.head_dim)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.n_embd)
        self.register_buffer('tril', torch.tril(torch.ones((config.ctx_len, config.ctx_len), dtype=torch.long)).view(1, 1, 1, config.ctx_len, config.ctx_len))
        self.rope = RotaryPositionEmbedding(config)

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embd

    def forward(self, x):
        B, T, C = x.shape

        query = self.q_proj(x).view(B, T, self.n_heads, self.head_dim) # (B, T, C) -> (B, T, n_heads * head_dim) -> (B, T, n_heads, head_dim)
        key = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim) # (B, T, C) -> (B, T, kv_heads * head_dim) -> (B, T, kv_heads, head_dim)
        value = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, T, C) -> (B, T, kv_heads * head_dim) -> (B, T, kv_heads, head_dim) -> (B, kv_heads, T, head_dim)

        query = self.rope(query).transpose(1, 2) # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        key = self.rope(key).transpose(1, 2) # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)

        query = query.view(B, self.n_heads // self.n_kv_heads, self.n_kv_heads, T, self.head_dim)
        key = key.unsqueeze(1).expand(-1, self.n_heads // self.n_kv_heads, -1, -1, -1)
        value = value.unsqueeze(1).expand(-1, self.n_heads // self.n_kv_heads, -1, -1 ,1)

        wei: Tensor = (query @ key.transpose(-2, -1)) / math.sqrt(self.n_heads) # (B, n_heads // kv_heads, kv_heads, T, head_dim) @ (B, n_heads // kv_heads, kv_heads, T, head_dim) -> (B, n_heads // kv_heads, kv_heads, T, T)
        wei = wei.masked_fill(self.tril[:, :, :, :T, :T] == 0, float('-inf'))
        wei = wei.softmax(-1)

        h_out = wei @ value # (B, n_heads // kv_heads, kv_heads, T, T) @ (B, n_heads // kv_heads, kv_heads, T, head_dim) -> (B, n_heads // kv_heads, kv_heads, T, head_dim)
        h_out = h_out.transpose(2, 3).contigous().view(B, self.n_heads // self.n_kv_heads, T, self.n_kv_heads * self.head_dim) # (B, n_heads // kv_heads, kv_heads, T, head_dim) -> (B, n_heads // kv_heads, T, kv_heads, head_dim) -> (B, n_heads // kv_heads, T, kv_heads * head_dim)
        h_out = h_out.transpose(1, 2)[:, :, 0, :] # (B, n_heads // kv_heads, T, kv_heads * head_dim) -> (B, T, n_heads // kv_heads, kv_heads * head_dim) -> (B, T, n_heads * head_dim)
        
        x_out = self.out_proj(h_out) # (B, T, n_heads * head_dim) -> (B, T, C)
        # TODO: dropout before returning?
        return x_out

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
