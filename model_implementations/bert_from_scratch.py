import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import AutoModelForMaskedLM, BertConfig, BertModel, BertTokenizer, load_tf_weights_in_bert


@dataclass
class skBERTConfig:
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12


class MultiHeadAttention(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init__()

        self.self = nn.ModuleDict(dict(
            query = nn.Linear(config.hidden_size, config.hidden_size),
            key = nn.Linear(config.hidden_size, config.hidden_size),
            value = nn.Linear(config.hidden_size, config.hidden_size)
        ))
        self.output = nn.ModuleDict(dict(
            dense = nn.Linear(config.hidden_size, config.hidden_size),
            LayerNorm = nn.LayerNorm(config.hidden_size)
        ))
        self.num_attention_heads = config.num_attention_heads

    def forward(self, x, attn_mask: Tensor = None):
        B, T, C = x.shape

        # (B, T, C) -> (B, T, num_attention_heads, head_size) -> (B, num_attention_heads, T, head_size)
        q = self.self.query(x).view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2)
        k = self.self.key(x).view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2)
        v = self.self.value(x).view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2)

        # (B, num_attention_heads, T, head_size) @ (B, num_attention_heads, head_size, T) -> (B, num_attention_heads, T, T)
        wei: Tensor = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(q.size(-1)))
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :]
            wei = wei.masked_fill(attn_mask == 0, float('-inf'))
        wei = wei.softmax(-1)
        out = wei @ v # (B, num_attention_heads, T, T) @ (B, num_attention_heads, T, head_size) -> (B, num_attention_heads, T, head_size)

        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, num_attention_heads, T, head_size) -> (B, T, num_attention_heads, head_size) -> (B, T, C == n_head*head_size)
        out = self.output.dense(out)
        out = self.output.LayerNorm(out)

        return out

class Block(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init__()

        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.ModuleDict(dict(
            dense = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        ))
        self.output = nn.ModuleDict(dict(
            dense = nn.Linear(4 * config.hidden_size, config.hidden_size),
            LayerNorm = nn.LayerNorm(config.hidden_size)
        ))

    def forward(self, x, attn_mask = None):
        x = x + self.attention(x, attn_mask)
        
        # Perform MLP
        mlp_x = self.intermediate.dense(x)
        mlp_x = F.gelu(mlp_x)
        mlp_x = self.output.dense(mlp_x)
        mlp_x = self.output.LayerNorm(mlp_x)
        
        x = x + mlp_x

        return x

class skBERT(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init__()

        self.embeddings = nn.ModuleDict(dict(
            word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size),
            position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size),
            token_type_embeddings = nn.Embedding(2, config.hidden_size),
            LayerNorm = nn.LayerNorm(config.hidden_size)
        ))
        self.encoder = nn.ModuleDict(dict(
            layer = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        ))
    
    def forward(self, x, attn_mask=None, token_type_ids=None):
        B, T = x.shape

        tok_embed = self.embeddings.word_embeddings(x)
        pos_embed = self.embeddings.position_embeddings(torch.arange(T, dtype=torch.long, device=x.device))
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x)
        token_type_embed = self.embeddings.token_type_embeddings(token_type_ids)
        
        x = tok_embed + pos_embed + token_type_embed
        x = self.embeddings.LayerNorm(x)

        for block in self.encoder.layer:
            x = block(x, attn_mask)
        
        return x # (B, T, C)


class BERTPredictionHeadLM(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.transform = nn.ModuleDict(dict(
            dense = nn.Linear(config.hidden_size, config.hidden_size),
            LayerNorm = nn.LayerNorm(config.hidden_size)
        ))
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform.dense(x)
        x = F.gelu(x)
        x = self.transform.LayerNorm(x)
        logits = self.decoder(x)
        return logits

class skBERTForMaskedLM(nn.Module):
    def __init__(self, config: skBERTConfig):
        super().__init__()
        self.bert = skBERT(config)
        self.cls = nn.ModuleDict(dict(
            predictions = BERTPredictionHeadLM(config),
        ))
        self.block_size = config.max_position_embeddings
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        config = BertConfig.from_pretrained(model_name_or_path)
        model = cls(config)

        state_dict = AutoModelForMaskedLM.from_pretrained(model_name_or_path).state_dict()
        model.load_state_dict(state_dict)

        return model
    def forward(self, x, attn_mask=None, token_type_ids=None, targets=None):
        x = self.bert(x, attn_mask=attn_mask, token_type_ids=token_type_ids) # (B, T) -> (B, T, C)
        logits = self.cls.predictions(x)
        loss = None

        if targets is not None:
            logits = logits[:, -1, :]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        
        return logits, loss

    def generate(self, input_tensor: Tensor, attn_mask: Tensor, token_type_ids: Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            input_tensor = 
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


model = skBERTForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
# model = skBERTForMaskedLM(skBERTConfig)
print('Woo!')
tokenizer: BertTokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

input_docs = [
    "I'm a language model, ",
    "I'm a language model, ",
    "I'm a language model, "
]
tokenized_inputs = tokenizer(input_docs)

input_toks = torch.tensor(tokenized_inputs['input_ids'])
attn_masks = torch.tensor(tokenized_inputs['attention_mask'])
token_type_ids = torch.tensor(tokenized_inputs['token_type_ids'])

x = model.generate(input_tensor=input_toks, attn_mask=attn_masks, token_type_ids=token_type_ids, max_new_tokens=50)

for gen in x:
    print(tokenizer.decode(gen))

# ------------------------------------ training loop ------------------------------------

class DataloaderLite():
    def __init__(self) -> None:
        pass