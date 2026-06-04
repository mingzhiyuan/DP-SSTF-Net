from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


def ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

class MiCR(nn.Module):
    def __init__(self, channels: int, k_size: int = 5):
        super().__init__()
        k = ensure_odd(max(1, int(k_size)))
        if channels <= 1:
            k = 1
        else:
            k = min(k, channels if (channels % 2 == 1) else (channels - 1))
            k = ensure_odd(max(1, k))
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=-1, keepdim=False)
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        w = torch.sigmoid(y).unsqueeze(-1)
        return x * w

class LSSBlock(nn.Module):
    def __init__(self, d: int, kernel_size: int = 7, dropout: float = 0.3, norm: str = "rms"):
        super().__init__()
        self.in_proj = nn.Linear(d, 2 * d, bias=False)
        self.dwconv = nn.Conv1d(d, d, kernel_size=kernel_size, padding=kernel_size // 2, groups=d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = RMSNorm(d) if norm == "rms" else nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u, v = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        y = u * torch.sigmoid(v)
        y = self.out_proj(y)
        y = self.drop(y)
        return self.norm(residual + y)


class LSSStack(nn.Module):
    def __init__(self, d: int, depth: int = 4, kernel_size: int = 7, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.ModuleList([LSSBlock(d, kernel_size=kernel_size, dropout=dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

class PKIFreqEncoder(nn.Module):
    def __init__(self, in_channels: int, d: int, kernels: Tuple[int, ...] = (3, 5, 7), dropout: float = 0.3):
        super().__init__()
        self.depthwise = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, k, padding=k // 2, groups=in_channels, bias=False)
            for k in kernels
        ])
        self.fuse_k = nn.Conv1d(in_channels * len(kernels), in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.mix = nn.Conv1d(in_channels, d, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        xs = [conv(P) for conv in self.depthwise]
        x = torch.cat(xs, dim=1)
        x = self.fuse_k(x)
        x = self.act(x)
        x = self.mix(x)
        x = self.drop(self.act(x))
        return x

class SoftHist1D(nn.Module):
    def __init__(self, bins: int = 16, sigma: float = 0.08, eps: float = 1e-6):
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        self.eps = eps
        self.register_buffer("centers", torch.linspace(0.0, 1.0, bins))

    def forward(self, x01: torch.Tensor):
        c = self.centers.view(1, self.bins, 1)
        x = x01.unsqueeze(1)
        s = torch.exp(-0.5 * ((x - c) / self.sigma) ** 2)
        s = s / (s.sum(dim=1, keepdim=True) + self.eps)
        h = s.mean(dim=-1)
        return s, h


class HistFreqAtt(nn.Module):
    def __init__(self, bins: int = 16, hidden: int = 64, sigma: float = 0.08, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hist = SoftHist1D(bins=bins, sigma=sigma, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(bins, hidden),
            nn.GELU(),
            nn.Linear(hidden, bins),
        )

    @staticmethod
    def minmax01(x: torch.Tensor, eps: float) -> torch.Tensor:
        mn = x.amin(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        return (x - mn) / (mx - mn + eps)

    def forward(self, S: torch.Tensor):
        q = S.abs().mean(dim=1)
        q01 = self.minmax01(q, self.eps)
        assign, hist = self.hist(q01)
        alpha = torch.softmax(self.mlp(hist), dim=-1)
        w = (alpha.unsqueeze(-1) * assign).sum(dim=1)
        w = w / (w.mean(dim=-1, keepdim=True) + self.eps)
        return S * w.unsqueeze(1), w

class ETTE(nn.Module):
    def __init__(self, d: int, kernel_size: int = 5, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(1, d, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = RMSNorm(d)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        B, C, F = P.shape
        x = P.reshape(B * C, 1, F)
        x = self.act(self.conv(x))
        x = x.mean(dim=-1)
        x = self.norm(self.drop(x))
        return x.view(B, C, -1)

class MiTPE(nn.Module):
    def __init__(self, in_channels: int, d: int, patch_size: int = 25, stride: int = 10):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels, d,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            bias=False
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.proj(x))
        return z.transpose(1, 2)

class GFU(nn.Module):
    def __init__(self, d: int, hidden: Optional[int] = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or max(32, d // 2)
        self.gate = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        out = g * a + (1.0 - g) * b
        return self.drop(out)


class SPGFMap1D(nn.Module):
    def __init__(self, d: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(2 * d, d, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, att: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.conv(torch.cat([att, raw], dim=1)))
        out = g * att + (1.0 - g) * raw
        return self.drop(out)


class SPGFToken(nn.Module):
    def __init__(self, d: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or max(32, d // 2)
        self.proj = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, att: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.proj(torch.cat([att, raw], dim=-1)))
        out = g * att + (1.0 - g) * raw
        return self.drop(out)


class DP_SSTFNet(nn.Module):
    def __init__(
        self,
        channels: int,
        d: int = 128,
        chan_depth: int = 3,
        temp_depth: int = 4,
        bins: int = 16,
        dropout: float = 0.4,
        patch_size: int = 25,
        patch_stride: int = 10,
        eca_kernel: int = 5,
        eps: float = 1e-6,
        token_fuse_dropout: float = 0.2,
        map_fuse_dropout: float = 0.2,
    ):
        super().__init__()
        self.eps = eps

        self.eca_freq = MiCR(channels=channels, k_size=eca_kernel)
        self.eca_time = MiCR(channels=channels, k_size=eca_kernel)

        self.spec_enc = PKIFreqEncoder(in_channels=channels, d=d, kernels=(3, 5, 7), dropout=dropout)
        self.spec_att = HistFreqAtt(bins=bins, hidden=max(64, d // 2), sigma=0.08, eps=eps)

        self.chan_tok = ETTE(d=d, kernel_size=5, dropout=dropout)
        self.chan_model = LSSStack(d=d, depth=chan_depth, kernel_size=5, dropout=dropout)

        self.temp_embed = MiTPE(in_channels=channels, d=d, patch_size=patch_size, stride=patch_stride)
        self.temp_model = LSSStack(d=d, depth=temp_depth, kernel_size=7, dropout=dropout)

        self.fuse_spec_map = SPGFMap1D(d=d, dropout=map_fuse_dropout)
        self.fuse_chan_tok = SPGFToken(d=d, dropout=token_fuse_dropout)
        self.fuse_time_tok = SPGFToken(d=d, dropout=token_fuse_dropout)

        self.gfu_freq = GFU(d=d, dropout=dropout)
        self.gfu_cross = GFU(d=d, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 2),
        )

    def rfft_logpower(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(x, dim=-1)
        return torch.log(X.abs().pow(2) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B, C, T), got {tuple(x.shape)}")

        P_raw = self.rfft_logpower(x)
        P_att = self.eca_freq(P_raw)

        tok_raw = self.chan_tok(P_raw)
        tok_att = self.chan_tok(P_att)
        tok_fused = self.fuse_chan_tok(tok_att, tok_raw)

        tok_fused = self.chan_model(tok_fused)
        u_freq = tok_fused.mean(dim=1)

        t_raw = self.temp_embed(x)
        x_att = self.eca_time(x)
        t_att = self.temp_embed(x_att)

        t_fused = self.fuse_time_tok(t_att, t_raw)
        t_fused = self.temp_model(t_fused)
        v_temp = t_fused.mean(dim=1)

        u = self.gfu_cross(u_freq, v_temp)
        return self.head(u)


