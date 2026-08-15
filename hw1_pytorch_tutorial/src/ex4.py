from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images to patch tokens."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = torch.einsum('b c h p w q -> b h w c p q', x).contiguous()
    return x.view(B, H // patch_size * W // patch_size, patch_size * patch_size * C)

# TODO: Add positional encoding as done in the ViT paper and patch projection
class PatchEmbed(nn.Module):
    def __init__(self, patch_dim: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.proj = nn.Linear(patch_dim * patch_dim * in_channels, d_model)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        return self.proj(x_patches)


class PositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_tokens + 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weights.unsqueeze(0)

# TODO: Define the variants you want to compare against each other from the GLU paper. Justify your choice.
class FeedForward(nn.Module):
    """
    Standard Transformer FFN:
      x -> Linear(d_model->d_ff) -> GELU -> Dropout -> Linear(d_ff->d_model) -> Dropout
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class GLUFeedForward(nn.Module):
    """GLU-family FFN"""
    def __init__(self, d_model: int, d_ff_gated: int, dropout: float, variant: str):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff_gated, bias=False)
        self.V = nn.Linear(d_model, d_ff_gated, bias=False)
        self.W2 = nn.Linear(d_ff_gated, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GEGLU and SwiGLU -->> best perplexities
        if self.variant == 'GLU':
            return self.W2(self.drop(F.sigmoid(self.W(x)) * self.V(x)))
        elif self.variant == 'GEGLU':
            return self.W2(self.drop(F.gelu(self.W(x)) * self.V(x)))
        elif self.variant == 'SwiGLU':
            return self.W2(self.drop(F.silu(self.W(x)) * self.V(x)))
        else:
            raise ValueError(f'Unknown variant {self.variant}')


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN encoder block:
      x = x + Dropout(SelfAttn(LN(x)))
      x = x + Dropout(MLP(LN(x)))
    """
    def __init__(self, d_model: int, n_heads: int, mlp: nn.Module, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = mlp
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0])
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TinyViT(nn.Module):
    """
    Tiny ViT-style classifier for MNIST.
    - patchify -> patch embed -> pos embed -> blocks -> mean pool -> head
    """
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        mlp_kind: str,
    ):
        super().__init__()
        assert 28 % patch_size == 0
        grid = 28 // patch_size
        self.num_tokens = grid * grid
        self.patch_size = patch_size
        patch_dim = patch_size

        self.patch_embed = PatchEmbed(patch_dim, d_model, in_channels=1)
        self.pos_embed = PositionalEmbedding(self.num_tokens, d_model)

        # Apply 2/3 rule from Shazeer for fair parameter count comparison
        if mlp_kind == 'FFN':
            mlp = FeedForward(d_model, d_ff, dropout)
        elif mlp_kind in ['GLU', 'GEGLU', 'SwiGLU']:
            d_ff_gated = int(d_ff * 2 / 3)  # 2/3 rule: GLU has 3 matrices vs FFN's 2
            mlp = GLUFeedForward(d_model, d_ff_gated, dropout, variant=mlp_kind)
        else:
            raise ValueError(f"Unknown mlp_kind: {mlp_kind}")

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp=mlp,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = patchify(x, self.patch_size)
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)
        return self.head(x)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 128
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.01
    device: str = "cpu"  # set "cuda" if available



def train_one_run(
    mlp_kind: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
) -> dict:
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses: list[float] = []
    test_accs: list[float] = []

    for epoch in range(cfg.epochs):

        # Train loop
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            
            print(xb.shape, yb.shape)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # Evaluation loop NOTE: Should be no need to change this
        model.eval()
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                logits = model(xb)
                correct += (logits.argmax(dim=-1) == yb).float().sum().item()
                total += yb.numel()

        test_accs.append(correct / total)
        print(f"[{mlp_kind}] epoch {epoch+1}/{cfg.epochs} | test acc: {test_accs[-1]:.4f}")

    return {
        "mlp_kind": mlp_kind,
        "train_losses": train_losses,
        "test_accs": test_accs,
    }


cfg = TrainConfig(seed=0, batch_size=128, epochs=5, lr=3e-4, weight_decay=0.01, device="cpu")

tfm = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

# Tiny model example. TODO: You're welcome to experiment with these parameters
patch_size = 4
d_model = 64
n_heads = 4
n_layers = 2
d_ff = 256
dropout = 0.1

runs = ["FFN", "GEGLU", "SwiGLU"]
results = []

for kind in runs:
    model = TinyViT(
        patch_size=patch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        mlp_kind=kind,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = train_one_run(kind, model, train_loader, test_loader, cfg)
    print(f"\nRun: {kind} | Number of parameters: {num_params} | Final test accuracy: {out['test_accs'][-1]:.4f} | Best test accuracy: {max(out['test_accs']):.4f}")
    results.append(out)
