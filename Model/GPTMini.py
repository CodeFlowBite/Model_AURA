
from dataclasses import dataclass
from typing import Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTMiniConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    block_size: int = 512  # longitud máxima del contexto
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTMiniConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        if attn_mask is not None:
            att = att + attn_mask

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.out(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTMiniConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTMiniConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTMini(nn.Module):
    def __init__(self, config: GPTMiniConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx, attn_mask=None):
        B, T = idx.size()
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(0, T, device=idx.device)).unsqueeze(0)
        x = self.drop(tok + pos)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)
        return self.head(x)

    def save(self, version_name: str):
        os.makedirs('versiones', exist_ok=True)
        path = f'versiones/{version_name}.pt'
        torch.save({'model_state': self.state_dict(), 'config': self.config.__dict__}, path)
        print(f"Modelo guardado en {path}")

    @staticmethod
    def load(version_name: str):
        path = f'versiones/{version_name}.pt'
        data = torch.load(path, map_location='cpu')
        cfg = GPTMiniConfig(**data['config'])
        model = GPTMini(cfg)
        model.load_state_dict(data['model_state'])
        print(f"Modelo cargado desde {path}")
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None, eos_token_id=2):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits = self.forward(idx_cond)[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                vals, _ = torch.topk(logits, top_k)
                min_vals = vals[:, -1].unsqueeze(1)
                logits[logits < min_vals] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
            if eos_token_id is not None and (next_token.squeeze() == eos_token_id).all():
                break
        return idx
