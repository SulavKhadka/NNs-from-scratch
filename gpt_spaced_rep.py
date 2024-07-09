import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

ctx_len = 16
batch_size = 4
n_embed = 32
head_size = 16
dropout_rate = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


## DATA PREP

# Open data file
with open("input.txt", "r") as file:
    text = file.read()

# calculate vocab and vocab_size
unique_chars = sorted(list(set(text)))
vocab = unique_chars
vocab_size = len(unique_chars)

# make tokenizer with encode and decode functions
stoi = {char: i for i, char in enumerate(vocab)}
itos = {i: char for i, char in enumerate(vocab)}

encode = lambda prompt: [stoi[char] for char in prompt]
decode = lambda tokens: [itos[i] for i in tokens]


# encode and split text into train and validation sets
encoded_text = encode(text)
split_idx = int(0.9 * len(encoded_text))
train_data = encoded_text[:split_idx]
val_data = encoded_text[split_idx:]

# write out a helper method to batch data for effecient GPU computation
def batch_data():
    sample_idxs = torch.randint(len(train_data) - ctx_len, (batch_size,))
    input_data = torch.stack([train_data[idx: idx+ctx_len] for idx in sample_idxs])
    expected_data = torch.stack([train_data[idx+1: idx+ctx_len+1] for idx in sample_idxs])
    
    input_data.to(device)
    expected_data.to(device)
    
    return input_data, expected_data


## GPT ARCH

# Encode prompt text with tokenizer
# Get n_embed dim vectors from tok_embed_table for n_ctx tokens
# Add that into the (n_ctx, n_embed) vectors from pos_embed_table
# Initialize a chain of N blocks (each's output feeding the next block as input)
# Pass n_ctx token vectors into transformers block
# Get the resulting vectors from the last block and pass it through LayerNorm
# Pass that through a Linear layer
# retrun the resulting logits as the final answer
# calculate loss if in the training loop (perform cross entropy on the output logits and the expected logits)

# Self-Attention Head
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        
        self.query = nn.Linear(n_embed, head_size)
        self.key = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer('tril', torch.tril(torch.ones((n_embed, n_embed))))
    
    def forward(self, x):
        B, T, C = x.shape

        # Given a vector of n_ctx embeddings pass them through a Query and Key linear layers to get head_size dim vectors for every token
        q = self.query(x) # (B, ctx_len, n_embed) -> (B, ctx_len, head_size)
        k = self.key(x) # (B, ctx_len, n_embed) -> (B, ctx_len, head_size)

        # perform a (Query @ Transpose(Key))/sqrt(head_size) dot product to get affinites of each token to every other token. (This is our wei vector)
        wei: Tensor = q @ torch.transpose(k, -2, -1) # (B, ctx_len, head_size) @ (B, head_size, ctx_len) -> (B, ctx_len, ctx_len)
        wei = wei * head_size ** -0.5
        # do a masked fill to only activate the lower triangle
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        # perfrom softmax over it to get normalized values for the per token affinites
        wei = wei.softmax(-1) # exponentiate and avg across embedding vector per token
        # dropout for large scale training (TODO: why?)
        wei = self.dropout(wei)

        # In the same manner as Query and Key calculate a Value vector from the token embeddings using a Linear layer
        v = self.value(x)

        # Perform wei @ Values with the resulting wei values from the Q @ K calculation to get the output vector of head_size
        out = wei @ v # (B, ctx_len, ctx_len) @ (B, ctx_len, head_size) -> (B, ctx_len, head_size)
        return out

        

# Multi Headed Attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads) -> None:
        super().__init__()
        # initalize num_heads Self-Attention heads each with a head_size = n_embed/num_heads (we need to make sure the divison gives a whole number)
        self.head_size = n_embed // num_heads
        self.attention_heads = [SelfAttentionHead(self.head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # For the input vector x, pass them through all heads and concat the results together to form a output embedding of n_embed dim. ("communication" step)
        out = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        # Pass the resulting vector through a linear layer so the outputs from different heads can interact.
        out = self.proj(out)
        # add dropout here
        out = self.dropout(out)
        # return resulting vector
        return out

# Feed Forward Network ("computation" step in the transformers block)
# simple Linear Layer
# followed by a Non-Linearity(Relu)

# Transformer Block
# given a input vector of token embeddings x:
# perform layerNorm (1)
# pass it through the multi head attention layer + original token embedding vectors x
# perform layerNorm on output of MHA (2)
# pass it through feedeforward
# return resulting token vectors

## Training Loop

# activate the optimizer (AdamW)
# get a batch of inputs and expected outputs from the training data
# pass it through the model to get logits and loss
# clear the prev optimizer state
# call loss.backward
# call optimizer.step to tweak param weights to minimize loss

## Generation

# For an input block of tokens
# loop N times (N being the num of tokens we want to generate)
# for each loop(generation step):
    # get the last n_ctx tokens as input token vector
    # pass it through the model to get logits
    # take the last token's logits from the output
    # Sample it with torch.multinomial(logits)
    # concat the sampled token's embedding into the block of tokens that was inputted
# return the resulting block of tokens
