import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import BertTokenizer

@dataclass
class skBERTConfig:
    vocab_size: 30522
    block_size: 512
    n_embd: 768
    n_layer: 12
    n_head: 12


class MultiHeadAttention(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init__()

        self.self = nn.ModuleDict(dict(
            query = nn.Linear(config.n_embd, config.n_embd),
            key = nn.Linear(config.n_embd, config.n_embd),
            value = nn.Linear(config.n_embd, config.n_embd)
        ))
        self.output = nn.ModuleDict(dict(
            dense = nn.Linear(config.n_embd, config.n_embd),
            LayerNorm = nn.LayerNorm(config.n_embd)
        ))
        self.n_head = config.n_head

    def forward(self, x, attn_mask: Tensor = None):
        B, T, C = x.shape

        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q = self.self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        wei: Tensor = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(q.size(-1)))
        if attn_mask is not None:
            wei = wei.masked_fill(attn_mask == 0, float('-inf'))
        wei = wei.softmax(-1)
        out = wei @ v # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)

        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C == n_head*head_size)
        out = self.output.dense(out)
        out = self.output.LayerNorm(out)

        return out

class Block(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init__()

        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.output = nn.ModuleDict(dict(
            dense = nn.Linear(4 * config.n_embd, config.n_embd),
            LayerNorm = nn.LayerNorm(config.n_embd)
        ))

    def forward(self, x, attn_mask = None):
        x = x + self.attention(x, attn_mask)
        
        # Perform MLP
        mlp_x = self.intermediate(x)
        mlp_x = F.gelu(mlp_x)
        mlp_x = self.output.dense(mlp_x)
        mlp_x = self.output.LayerNorm(mlp_x)
        
        x = x + mlp_x

        return x

class skBERT(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init()

        self.embeddings = nn.ModuleDict(dict(
            word_embeddings = nn.Embedding(config.vocab_size, config.n_embd),
            position_embeddings = nn.Embedding(config.block_size, config.n_embd),
            token_type_embeddings = nn.Embedding(2, config.n_embd),
            LayerNorm = nn.LayerNorm(config.n_embd)
        ))
        self.encoder = nn.ModuleDict(dict(
            layer = nn.ModuleList([Block(config) for _ in config.n_layer])
        ))
    
    def forward(self, x, attn_mask=None, token_type_ids=None):
        B, T = x.shape

        tok_embed = self.embeddings.word_embeddings(x)
        pos_embed = self.embeddings.position_embeddings(torch.arange(T, dtype=torch.long))
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x)
        token_type_embed = self.embeddings.token_type_embeddings(token_type_ids)
        
        inputs = tok_embed + pos_embed + token_type_embed
        inputs = self.embeddings.LayerNorm(inputs)

        for block in self.encoder.layer:
            x = block(x, attn_mask)
        
        return x # (B, T, C)


class skBERTForMaskedLM(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init__()

        self.bert = skBERT(config)
        self.cls = nn.ModuleDict(dict(
            predictions = nn.ModuleDict(dict(
                bias = nn.Parameter(torch.zeros(config.n_embd)),
                transform = nn.ModuleDict(dict(
                    dense = nn.Linear(config.n_embd, config.n_embd),
                    LayerNorm = nn.LayerNorm(config.n_embd)
                ))
            )),
            decoder = nn.Linear(config.n_embd, config.vocab_size)
        ))

        self.cls.decoder.bias = self.cls.predictions.bias
        self.block_size = config.block_size
    
    def forward(self, x, attn_mask=None, token_type_ids=None, targets=None):
        B, T = x.shape

        x = self.bert(x, attn_mask=attn_mask, token_type_ids=token_type_ids) # (B, T) -> (B, T, C)
        x = self.cls.transform.dense(x)
        x = F.gelu(x)
        x = self.cls.transform.LayerNorm(x)

        logits = self.decoder(x)
        loss = None

        if targets is not None:
            logits = logits[:, -1, :]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        
        return logits, loss

    def generate(self, input_tensor: Tensor, attn_mask: Tensor, token_type_ids: Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            input_cropped = input_tensor[:, -self.block_size:]

            logits, loss = self(input_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            new_tok = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat((input_tensor, new_tok), dim=-1)
        return input_tensor

def mask_inputs_for_mlm(inputs: Tensor, tokenizer: BertTokenizer, mlm_probability: float = 0.15):
    labels: Tensor = inputs.clone()

    mask_prob_matrix = torch.full(inputs.shape, mlm_probability)
    masked_indices = torch.bernoulli(mask_prob_matrix).bool()

    mask_token_indices = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
    inputs[mask_token_indices] = tokenizer.mask_token_id

    rand_token_indices = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~mask_token_indices
    rand_tokens = torch.randint(len(tokenizer), inputs.shape, dtype=torch.long)
    inputs[rand_token_indices] = rand_tokens[rand_token_indices]

    return inputs, labels


model = skBERTForMaskedLM(skBERTConfig)

tokenizer: BertTokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

input_docs = [
    "I'm a language model, ",
    "I'm a language model, ",
    "I'm a language model, "
]
tokenized_inputs = tokenizer(input_docs)

input_toks = tokenized_inputs['input_ids']
attn_masks = tokenized_inputs['attention_mask']
token_type_ids = tokenized_inputs['token_type_ids']

x = model.generate(input_tensor=input_toks, attn_mask=attn_masks, token_type_ids=token_type_ids)
