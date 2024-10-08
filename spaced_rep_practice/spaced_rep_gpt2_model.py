import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import math
import time
from dataclasses import dataclass
import inspect

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # 50,000 BPE iterations + 256 unicode bytes + 1 special token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)
                                                .view(1, 1, config.block_size, config.block_size)))
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_embd = config.n_embd
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.shape

        exploded_attn: Tensor = self.c_attn(x)
        q, k, v = exploded_attn.split(self.n_embd, dim=-1)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        # wei: Tensor = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)) 
        wei = torch.einsum('bthq, bThq -> bhtT', [q.view(B, T, self.n_head, C // self.n_head), k.view(B, T, self.n_head, C // self.n_head)])
        
        wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # cant forget to crop the bias mask to T(num_tokens) of the current input x
        wei = wei.softmax(dim=-1)
        wei = self.attn_dropout(wei)

        # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        # out: Tensor = wei @ v 
        out = torch.einsum('bqtT, bTqh -> bqth', [wei, v])
        
        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x) # (B, T, C) -> (B, T, 4 * C)
        x = self.gelu(x)
        x = self.c_proj(x) # (B, T, 4 * C) -> (B, T, C)
        x = self.dropout(x)

        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight # weight typing as followed in the original GPT-2 implementation. pop quiz: but why?

        self.apply(self._init_weights) # apply weight initializtion to every module with logic described in _init_weights()

    def _init_weights(self, module: nn.Module):
        # if linear layer apply std dev of 0.02 and mean of 0 | if bias then initialize those with 0
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = std * (1 / math.sqrt(2*self.config.n_layer))
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # if embedding layer apply std dev of 0.02 and mean of 0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def _configure_optimizers(self, weight_decay, learning_rate, device):
        # getting a dict(parameter_name: parameters) of params that requires gradient updates
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # we want to apply the weight decay to layers/groups that are going to be matmul'ed not to the 1D scalars(is that the right word?)
        decay_group = [p for _, p in param_dict.items() if p.dim() >= 2]
        non_decay_group = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_group, 'weight_decay': weight_decay},
            {'params': non_decay_group, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=use_fused, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


    def forward(self, x, targets=None):
        B, T = x.shape

        tok_embd = self.transformer.wte(x) # (B, T) -> (B, T, C)
        pos_embd = self.transformer.wpe(torch.arange(T, dtype=torch.long, device=x.device)) # (T) -> (T, C)
        
        x = tok_embd + pos_embd # (B, T, C) + (T, C) -> (B, T, C)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # (B, T, C) -> (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def generate(self, idx, max_new_tokens: int):
        for i in range(max_new_tokens):
            idx_crop = idx[:, -self.config.block_size:]
            logits, loss = self(idx_crop)

            logits: Tensor = logits[:, -1, :]
            probs = logits.softmax(dim=-1)

            topk_probs, topk_idx = torch.topk(probs, k=50)
            new_sample = torch.multinomial(topk_probs, num_samples=1)

            new_tok = torch.gather(topk_idx, -1, new_sample)
            idx = torch.cat((idx, new_tok), dim=-1)
        return idx

# ---------------------------------------------- Dataloader ----------------------------------------------

class DataloaderLite():
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        self.curr_pos = 0

    def get_batch(self):
        batch_tokens = self.tokens[self.curr_pos: self.curr_pos + (self.B * self.T) + 1]
        x = batch_tokens[:-1].view(self.B, self.T)
        y = batch_tokens[1:].view(self.B, self.T)

        self.curr_pos += self.B * self.T
        if self.curr_pos + self.B * self.T + 1 > len(self.tokens):
            self.curr_pos = 0
        
        return x, y


# ---------------------------------------------- Inference ----------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")


# model = GPT.from_pretrained('gpt2')
# model.to(device)

# enc = tiktoken.get_encoding('gpt2')
# encoded_text = enc.encode("I am a language model,")
# batch = [encoded_text for _ in range(5)]
# tokens = torch.tensor(batch, device=device)

# x = model.generate(tokens, max_new_tokens=50)

# gen_text = enc.decode_batch(x.tolist())
# for i in gen_text:
#     print(i)

# ---------------------------------------------- Training Loop ----------------------------------------------

torch.set_float32_matmul_precision('high')
model: GPT = GPT(GPTConfig)
model = model.to(device)
model = torch.compile(model)

total_batch_tokens = 524288 # 2**19
B = 8
T = 1024
assert total_batch_tokens % (B*T) == 0 
grad_accum_steps = total_batch_tokens // (B * T)
print(f"total batch tokens: {total_batch_tokens} | grad accum steps: {grad_accum_steps}")

num_steps = 50
dataloader = DataloaderLite(B, T)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

for step in range(num_steps):
    t0 = time.time()
    optim.zero_grad()
    loss_accum = 0.0
    for i in range(grad_accum_steps):
        x, y = dataloader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # applying auto mixed precision to the layers that take it
            logits, loss = model(x, targets=y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    torch.cuda.synchronize()
    t1 = time.time()

    print(f"step {step}: loss = {loss_accum:.3f} | time = {(t1 - t0)*1000:.4f} | tok/sec = {(dataloader.B*dataloader.T*grad_accum_steps)/(t1 - t0):.3f}")
