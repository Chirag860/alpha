"""Optional shared-weight causal TCN trunk (§4.3).

Lower priority than in the crypto setup: at a 15-minute horizon on ~1-minute bars a
sequence has only ~30 steps of moderately informative history, so there is less for a
sequence model to do. When enabled it is used as a **representation learner**: train the
TCN on the cross-sectional rank target, then feed its penultimate embedding into the GBDT
as extra features. Extract embeddings **out-of-fold** (§4.3) -- the harness is responsible
for that; this module just fits/transforms.

Strictly causal by construction (left-padded dilated convolutions), so no future step
leaks into the current representation. A learned per-stock embedding lets the pooled model
capture idiosyncratic behavior without fragmenting the sample (§4.1).

Import of torch is guarded: :func:`torch_available` lets callers skip gracefully.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn

    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


def torch_available() -> bool:
    return _HAVE_TORCH


if _HAVE_TORCH:

    class _CausalConv1d(nn.Module):
        """Left-padded (causal) 1-D convolution."""

        def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = nn.functional.pad(x, (self.pad, 0))
            return self.conv(x)

    class _TCNBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int,
                     dropout: float) -> None:
            super().__init__()
            self.conv1 = _CausalConv1d(c_in, c_out, kernel, dilation)
            self.conv2 = _CausalConv1d(c_out, c_out, kernel, dilation)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

        def forward(self, x):
            y = self.drop(self.relu(self.conv1(x)))
            y = self.drop(self.relu(self.conv2(y)))
            res = x if self.down is None else self.down(x)
            return self.relu(y + res)

    class _TCNNet(nn.Module):
        def __init__(self, n_features: int, n_stocks: int, cfg) -> None:
            super().__init__()
            ch = int(cfg.channels)
            self.stock_emb = nn.Embedding(n_stocks, int(cfg.stock_embedding_dim))
            in_ch = n_features + int(cfg.stock_embedding_dim)
            blocks = []
            c_prev = in_ch
            for i in range(int(cfg.levels)):
                blocks.append(_TCNBlock(c_prev, ch, int(cfg.kernel_size), 2 ** i,
                                        float(cfg.dropout)))
                c_prev = ch
            self.tcn = nn.Sequential(*blocks)
            self.embed = nn.Linear(ch, int(cfg.embedding_dim))
            self.head = nn.Linear(int(cfg.embedding_dim), 1)

        def forward(self, x, stock_idx, return_embedding: bool = False):
            # x: (B, T, F); stock_idx: (B,)
            b, t, _ = x.shape
            se = self.stock_emb(stock_idx).unsqueeze(1).expand(b, t, -1)
            h = torch.cat([x, se], dim=-1).transpose(1, 2)   # (B, C, T)
            h = self.tcn(h)[:, :, -1]                          # last (causal) step
            emb = torch.relu(self.embed(h))
            if return_embedding:
                return emb
            return self.head(emb).squeeze(-1)


class TCNEmbedder:
    """Fit a causal TCN on the rank target and extract a penultimate embedding (§4.3).

    Sequences are per-name windows of the (already normalized) feature vector. Use
    :meth:`transform` to get an ``(N, embedding_dim)`` matrix to concatenate onto the GBDT
    features. Requires PyTorch; check :func:`torch_available` first.
    """

    def __init__(self, cfg, n_features: int, n_stocks: int) -> None:
        if not _HAVE_TORCH:  # pragma: no cover
            raise RuntimeError("PyTorch not available; TCNEmbedder cannot be constructed")
        self.cfg = cfg
        self.seq_len = int(cfg.seq_len)
        self.emb_dim = int(cfg.embedding_dim)
        self.net = _TCNNet(n_features, n_stocks, cfg)

    def _make_sequences(self, X: np.ndarray, stock_idx: np.ndarray,
                        day_idx: np.ndarray) -> tuple:
        """Build causal windows that never cross a (stock, day) boundary."""
        T = self.seq_len
        seqs, sidx, rows = [], [], []
        n = len(X)
        for i in range(n):
            lo = i - T + 1
            if lo < 0:
                continue
            if stock_idx[lo] != stock_idx[i] or day_idx[lo] != day_idx[i]:
                continue
            seqs.append(X[lo:i + 1])
            sidx.append(stock_idx[i])
            rows.append(i)
        if not seqs:
            return None, None, None
        return (np.asarray(seqs, np.float32), np.asarray(sidx, np.int64),
                np.asarray(rows, np.int64))

    def fit(self, X: np.ndarray, y: np.ndarray, stock_idx: np.ndarray,
            day_idx: np.ndarray, sample_weight: np.ndarray | None = None) -> "TCNEmbedder":
        seqs, sidx, rows = self._make_sequences(np.asarray(X, np.float32),
                                                stock_idx, day_idx)
        if seqs is None:  # pragma: no cover - tiny data
            return self
        yv = np.asarray(y, np.float32)[rows]
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg.lr))
        loss_fn = torch.nn.MSELoss()
        xt = torch.from_numpy(seqs)
        st = torch.from_numpy(sidx)
        yt = torch.from_numpy(yv)
        bs = int(self.cfg.batch_size)
        self.net.train()
        for _ in range(int(self.cfg.epochs)):
            perm = torch.randperm(len(xt))
            for b in range(0, len(xt), bs):
                idx = perm[b:b + bs]
                opt.zero_grad()
                pred = self.net(xt[idx], st[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
        return self

    def transform(self, X: np.ndarray, stock_idx: np.ndarray,
                  day_idx: np.ndarray) -> np.ndarray:
        """Return an ``(N, embedding_dim)`` matrix aligned to the input rows.

        Rows without a full causal window inherit a zero embedding (warm-up).
        """
        out = np.zeros((len(X), self.emb_dim), np.float32)
        seqs, sidx, rows = self._make_sequences(np.asarray(X, np.float32),
                                                stock_idx, day_idx)
        if seqs is None:  # pragma: no cover
            return out
        self.net.eval()
        with torch.no_grad():
            emb = self.net(torch.from_numpy(seqs), torch.from_numpy(sidx),
                           return_embedding=True).numpy()
        out[rows] = emb
        return out
