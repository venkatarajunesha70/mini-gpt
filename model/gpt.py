import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(
            embed_dim,
            3 * embed_dim
        )

        self.proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length
                )
            ).view(
                1,
                1,
                context_length,
                context_length
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        attention_scores = attention_scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        output = attention_weights @ v

        output = output.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        return self.proj(output)

class MLP(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                embed_dim,
                4 * embed_dim
            ),

            nn.GELU(),

            nn.Linear(
                4 * embed_dim,
                embed_dim
            )
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.attention = CausalSelfAttention(
            embed_dim,
            num_heads,
            context_length
        )

        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(embed_dim)

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x

class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length=128,
        embed_dim=256,
        num_heads=4,
        num_layers=4
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            context_length,
            embed_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                context_length
            )
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(
            embed_dim
        )

        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids, targets=None):

        B, T = input_ids.shape

        positions = torch.arange(
            T,
            device=input_ids.device
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss