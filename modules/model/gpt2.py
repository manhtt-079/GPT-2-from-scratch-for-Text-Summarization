import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.functional import Tensor
from math import sqrt

from config.config import ModelArgs


# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, temperature: int, prop: float = 0.1) -> None:
#         super().__init__()

#         self.temperature = temperature
#         self.dropout = nn.Dropout(p=prop)

#     def forward(self, q, k, v, mask=None) -> Dict[Tensor, Tensor]:
#         scores = torch.bmm(q, k.transpose(0, 2, 1))/sqrt(self.temperature)

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         weights = F.softmax(scores, dim=-1)

#         return torch.bmm(weights, v), weights

def scaled_dot_product(query: Tensor,
                       key: Tensor,
                       value: Tensor,
                       heads: int,
                       mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    dim_k = query.size(-1)

    scores = torch.bmm(query, key.transpose(1, 2))/sqrt(dim_k)
    if mask is not None:
        mask_repeat_size = (heads,) + tuple(1 for _ in range(mask.dim()-1))
        mask = mask.repeat(mask_repeat_size)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)

    return torch.bmm(weights, value), weights


class MultiheadAttention(nn.Module):
    def __init__(self, 
                 n_embeds: int,
                 n_heads: int,
                 ) -> None:
        super(MultiheadAttention, self).__init__()

        self.n_embeds = n_embeds
        self.n_heads = n_heads
        self.head_dim = self.n_embeds//self.n_heads
        self.q = nn.Linear(self.n_embeds, self.n_heads*self.head_dim)
        self.k = nn.Linear(self.n_embeds, self.n_heads*self.head_dim)
        self.v = nn.Linear(self.n_embeds, self.n_heads*self.head_dim)

        self.fc = nn.Linear(self.n_embeds, self.n_embeds)

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                values: Tensor,
                mask=None) -> Tuple[Tensor, Tensor]:

        batch_size, seq_len, embed_dim = keys.size()
        assert embed_dim == self.n_embeds, f"Input embedding dim ({embed_dim}) must match layer embedding dim {self.n_embeds}"

        queries, keys, values = self._split_head(queries, keys, values, batch_size, seq_len)

        # compute scaled dot-product attention
        queries = queries.transpose(1, 2).contiguous().view(batch_size*self.n_heads, seq_len, self.head_dim)
        keys = keys.transpose(1, 2).contiguous().view(batch_size*self.n_heads, seq_len, self.head_dim)
        values = values.transpose(1, 2).contiguous().view(batch_size*self.n_heads, seq_len, self.head_dim)

        # attention size of: [batch_size*n_heads, seq_len, head_dim]
        # weights size of: [batch_size*n_heads, seq_len, seq_len]
        attn, weights = scaled_dot_product(queries, keys, values, self.n_heads, mask)

        attn, weights = self._merge_head(attn, weights, batch_size, seq_len)

        return self.fc(attn), weights

    def _split_head(self,
                    queries: Tensor,
                    keys: Tensor,
                    values: Tensor,
                    batch_size: int,
                    seq_len: int) -> Tuple[Tensor, Tensor, Tensor]:

        queries = queries.view(batch_size, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.n_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.n_heads, self.head_dim)

        return queries, keys, values

    def _merge_head(self,
                    attn: Tensor,
                    weights: Tensor,
                    batch_size: int,
                    seq_len: int) -> Tuple[Tensor, Tensor]:

        attn = attn.view(batch_size, self.n_heads, seq_len, self.head_dim).transpose(1, 2)
        attn = attn.contiguous().view(batch_size, seq_len, self.n_heads*self.head_dim)

        weights = weights.view(batch_size, self.n_heads, seq_len, seq_len)

        return attn, weights


class FeedForward(nn.Module):
    def __init__(self, 
                 activation_func: str,
                 n_embeds: int,
                 n_inners: int,
                 residual_dropout: float) -> None:
        super(FeedForward, self).__init__()

        if activation_func == 'gelu':
            atv_func = nn.GELU()
        elif activation_func == 'silu':
            atv_func = nn.SiLU()
        elif activation_func == 'relu':
            atv_func = nn.ReLU()
        else:
            atv_func=nn.Tanh()
        
        self.fc = nn.Sequential(
            nn.Linear(n_embeds, n_inners),
            atv_func,
            nn.Linear(n_inners, n_embeds)
        )

        self.dropout = nn.Dropout(p=residual_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc(hidden_states)

        return self.dropout(hidden_states)


class DecoderLayer(nn.Module):
    def __init__(self, 
                 activation_func: str,
                 n_embeds: int,
                 n_inners: int,
                 n_heads: int,
                 residual_dropout: float,
                 layer_norm_epsilon: float
                 ) -> None:
        super(DecoderLayer, self).__init__()

        self.mask_mha = MultiheadAttention(n_embeds=n_embeds, n_heads=n_heads)
        self.fc = FeedForward(activation_func=activation_func,
                              n_embeds=n_embeds,
                              n_inners=n_inners,
                              residual_dropout=residual_dropout)
        
        self.ln_1 = nn.LayerNorm(n_embeds, eps=layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(n_embeds, eps=layer_norm_epsilon)

    def forward(self,
                hidden_state: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        attns, weights = self.mask_mha(hidden_state, hidden_state, hidden_state, mask)
        hidden_state = hidden_state + attns
        hidden_state = self.ln_1(hidden_state)

        hidden_state = hidden_state + self.fc(hidden_state)
        return self.ln_2(hidden_state), weights


class Embedding(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 n_embeds: int,
                 n_positions: int,
                 embed_dropout: float,
                 layer_norm_epsilon: float) -> None:
        super(Embedding, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, n_embeds)
        self.position_embeddings = nn.Embedding(n_positions, n_embeds)
        self.dropout = nn.Dropout(p=embed_dropout)
        self.ln = nn.LayerNorm(n_embeds, layer_norm_epsilon)

    def forward(self, x: Tensor) -> Tensor:
        pos_id = torch.arange(x.size(1), dtype=torch.long, device=torch.device(x.device)).unsqueeze(0)
        token_embed = self.token_embeddings(x)
        pos_embed = self.position_embeddings(pos_id)
        embed = token_embed + pos_embed
        embed = self.ln(embed)

        return self.dropout(embed)


class Decoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.embeddings = Embedding(vocab_size=config.vocab_size,
                                    n_embeds=config.n_embeds,
                                    n_positions=config.n_positions,
                                    embed_dropout=config.embed_dropout,
                                    layer_norm_epsilon=config.layer_norm_epsilon)
        
        self.layers = nn.ModuleList([DecoderLayer(activation_func=config.activation_func,
                                                  n_embeds=config.n_embeds,
                                                  n_inners=config.n_inners,
                                                  n_heads=config.n_heads,
                                                  residual_dropout=config.residual_dropout,
                                                  layer_norm_epsilon=config.layer_norm_epsilon)
                                     for _ in range(config.n_layers)])

    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        h_state = self.embeddings(x)
        for layer in self.layers:
            h_state, weights = layer(h_state, mask)

        return h_state, weights


class GPT2Model(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super(GPT2Model, self).__init__()

        self.decoder = Decoder(config)
        self.fc = nn.Linear(config.n_embeds, config.vocab_size)

    def _mask_pad_idx(self,
                      input_ids: Tensor,
                      pad_idx: int) -> Tensor:

        return (input_ids != pad_idx).unsqueeze(1)

    def forward(self,
                input_ids: Tensor,
                pad_idx: int) -> Tuple[Tensor, Tensor]:
        
        mask_pad = self._mask_pad_idx(input_ids, pad_idx)

        seq_len = input_ids.size(-1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=torch.device(input_ids.device))).view(1, seq_len, seq_len).bool()
        mask = mask & mask_pad
        h_state, weights = self.decoder(input_ids, mask)

        logits = self.fc(h_state)

        return logits, weights