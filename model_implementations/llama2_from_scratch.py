import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from dataclasses import dataclass
import math
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor

@dataclass
class Llama2ConfigSK:
    vocab_size: int = 32_000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 16

    batch_size: int = 2
    n_embed: int = 4096

    device: str = 'cuda'

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, config: Llama2ConfigSK):
        super().__init__()

        self.head_dim = config.hidden_size // config.num_attention_heads

        # compute the theta_freqs since they are constant vectors applied to any given input x.
        theta_numerator = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        thetas = 1 / torch.pow(10_000, theta_numerator / self.head_dim).to(config.device)
        m = torch.arange(0, config.n_embed).to(config.device)
        
        freqs = torch.einsum('i, j -> ij', [m, thetas]).float()
        self.freqs_complex = torch.polar(torch.ones_like(freqs), freqs).to(config.device)

    def forward(self, x, start_pos: int):
        # x.shape = (B, ctx_len, n_heads, head_dim) where n_heads can be either [n_heads, n_kv_heads]
        x_mod = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_mod) # (B, ctx_len, n_heads, head_dim // 2, 2) -> (B, ctx_len, n_heads, head_dim // 2)

        freqs_complex = self.freqs_complex[start_pos:start_pos+x.size(1)]
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # (ctx_len, head_dim // 2) -> (1, ctx_len, head_dim // 2) -> (1, ctx_len, 1, head_dim // 2)
        x_rotated = x_complex * freqs_complex

        x_out = torch.view_as_real(x_rotated) #(B, ctx_len, n_heads, head_dim // 2) -> (B, ctx_len, n_heads, head_dim // 2, 2)
        x_out = x_out.reshape(*x.shape) # (B, ctx_len, n_heads, head_dim // 2, 2) -> (B, ctx_len, n_heads, head_dim)
        return x_out

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Llama2ConfigSK):
        super().__init__()

        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones((config.n_embed, config.n_embed), dtype=torch.long)).view(1, 1, 1, config.n_embed, config.n_embed))
        self.rope = RotaryPositionEmbedding(config)

        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_embed = config.hidden_size

        self.k_cache = torch.zeros((config.batch_size, config.n_embed, config.num_key_value_heads, self.head_dim), device=config.device)
        self.v_cache = torch.zeros((config.batch_size, config.n_embed, config.num_key_value_heads, self.head_dim), device=config.device)

    def forward(self, x, start_pos: int):
        B, T, C = x.shape # (B, 1, C)

        query = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).to(torch.float16) # (B, 1, C) -> (B, 1, n_heads * head_dim) -> (B, 1, n_heads, head_dim)
        key = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).to(torch.float16) # (B, 1, C) -> (B, 1, kv_heads * head_dim) -> (B, 1, kv_heads, head_dim)
        value = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).to(torch.float16) # (B, 1, C) -> (B, 1, kv_heads * head_dim) -> (B, 1, kv_heads, head_dim)

        query = self.rope(query, start_pos) # (B, 1, n_heads, head_dim)
        key = self.rope(key, start_pos) # (B, 1, kv_heads, head_dim)

        self.k_cache[:B, start_pos:start_pos + T] = key
        self.v_cache[:B, start_pos:start_pos + T] = value

        key = self.k_cache[:B, :start_pos + T] # (B, 1, n_heads, head_dim) -> (B, ctx_len, n_heads, head_dim)
        value = self.v_cache[:B, :start_pos + T] # (B, 1, n_heads, head_dim) -> (B, ctx_len, n_heads, head_dim)

        query = query.transpose(1, 2) # (B, 1, n_heads, head_dim) -> (B, n_heads, 1, head_dim)
        key = key.transpose(1, 2) # (B, ctx_len, n_heads, head_dim) -> (B, n_heads, ctx_len, head_dim)
        value = value.transpose(1, 2) # (B, ctx_len, n_heads, head_dim) -> (B, n_heads, ctx_len, head_dim)

        query = query.view(B, self.n_heads // self.n_kv_heads, self.n_kv_heads, T, self.head_dim)
        key = key.unsqueeze(1).expand(-1, self.n_heads // self.n_kv_heads, -1, -1, -1)
        value = value.unsqueeze(1).expand(-1, self.n_heads // self.n_kv_heads, -1, -1 ,-1)

        wei: Tensor = (query @ key.transpose(-2, -1)) / math.sqrt(self.n_heads) # (B, n_heads // kv_heads, kv_heads, 1, head_dim) @ (B, n_heads // kv_heads, kv_heads, ctx_len, head_dim) -> (B, n_heads // kv_heads, kv_heads, 1, ctx_len)
        # wei = wei.masked_fill(self.tril[:, :, :, :T, :T] == 0, float('-inf'))
        wei = wei.softmax(-1)

        h_out = wei @ value # (B, n_heads // kv_heads, kv_heads, 1, ctx_len) @ (B, n_heads // kv_heads, kv_heads, ctx_len, head_dim) -> (B, n_heads // kv_heads, kv_heads, 1, head_dim)
        h_out = h_out.transpose(2, 3).contigous().view(B, self.n_heads // self.n_kv_heads, T, self.n_kv_heads * self.head_dim) # (B, n_heads // kv_heads, kv_heads, 1, head_dim) -> (B, n_heads // kv_heads, 1, kv_heads, head_dim) -> (B, n_heads // kv_heads, 1, kv_heads * head_dim)
        h_out = h_out.transpose(1, 2)[:, :, 0, :] # (B, n_heads // kv_heads, 1, kv_heads * head_dim) -> (B, 1, n_heads // kv_heads, kv_heads * head_dim) -> (B, 1, n_heads * head_dim)
        
        x_out = self.out_proj(h_out) # (B, 1, n_heads * head_dim) -> (B, 1, C)
        # TODO: dropout before returning?
        return x_out

class MLP(nn.Module):
    def __init__(self, config: Llama2ConfigSK) -> None:
        super().__init__() 

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = F.silu(self.up_proj(x)) * self.gate_proj(x)
        x = self.down_proj(x)
        # TODO: another dropout here?
        return x

class Block(nn.Module):
    def __init__(self, config: Llama2ConfigSK):
        super().__init__()

        self.self_attn = GroupedQueryAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
    
    def forward(self, x, start_pos: int):
        # x.shape = B, 1, C
        x = x + self.self_attn(self.input_layernorm(x), start_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        # TODO: probably a dropout here
        return x

class Llama2Model(nn.Module):
    def __init__(self, config: Llama2ConfigSK):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) 
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size)

    def forward(self, x: Tensor, start_pos: int):
        # x.shape = (B, 1)
        assert x.size(-1) == 1, "cant be more than 1 token at a time"
        
        x = self.embed_tokens(x) # (B, 1) -> (B, 1, C)
        for block in self.layers:
            x = block(x, start_pos)
        x = self.norm(x)

        return x

class LlamaCausalLM(nn.Module):
    def __init__(self, config: Llama2ConfigSK) -> None:
        super().__init__()

        self.model = Llama2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.max_ctx_len = config.n_embed

    @staticmethod
    def build(model_name_or_path: str, load_model: bool):
        prev_time = time.time()
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path)
        
        params = LlamaConfig.from_pretrained(model_name_or_path)

        model_args: Llama2ConfigSK = Llama2ConfigSK(
            vocab_size = params.vocab_size,
            hidden_size = params.hidden_size,
            intermediate_size = params.intermediate_size,
            
            num_hidden_layers = params.num_hidden_layers,
            num_attention_heads = params.num_attention_heads,
            num_key_value_heads = params.num_key_value_heads,

            n_embed = params.max_position_embeddings,
            device = 'cuda'
        )
        
        # if model_args.device == "cuda":
        #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # else:
        #     torch.set_default_tensor_type(torch.BFloat16Tensor)
        torch.set_default_dtype(torch.bfloat16)
        
        model = LlamaCausalLM(model_args).to(model_args.device)

        if load_model:
            hf_state_dict = hf_model.state_dict()
            model.load_state_dict(hf_state_dict, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        del hf_model
        return model

    def forward(self, x, start_pos, targets = None):
        # x.shape = (B, 1)
        x = self.model(x, start_pos) # (B, 1) -> (B, 1, n_embed)
        logits = self.lm_head(x) # (B, 1, n_embed) -> (B, 1, vocab_size)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        # idx.shape = (B, T) <- the T in this case is the len of the longest prompt in the batch and the rest come in padded to that length
        B, T = idx.shape
        generation_iters = min(self.max_ctx_len, T + max_new_tokens)
        for i in range(1, generation_iters):
            logits, loss = self(idx[:, i-1:i], i)
            pass
            


hf_model_path = "NousResearch/Llama-2-7b-hf"


llama_tok = LlamaTokenizer.from_pretrained(hf_model_path)
prompts = [ "Hello there today is " ]
tok_prompts = llama_tok(prompts, add_special_tokens=False)['input_ids']
final_prompts = []
for p in tok_prompts:
    prompt = [llama_tok.bos_token_id]
    prompt.extend(p)
    final_prompts.append(prompt)


model = LlamaCausalLM.build(hf_model_path, load_model=True)
print("We did it gang")

generated_toks = model.generate(torch.tensor(final_prompts, device='cuda'), max_new_tokens=50)

print(generated_toks)