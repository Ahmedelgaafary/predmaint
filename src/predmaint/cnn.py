"""1D-CNN baseline on raw vibration windows (PyTorch). Optional: `pip install predmaint[deep]`.

Architecture: wide first kernel with stride (learns a filter bank, as in WDCNN-style bearing models), three
small conv blocks, global average pooling. Each window is z-scored, so the network sees waveform shape only:
no shaft speed, no bearing geometry. That makes it a fair "no domain knowledge" deep-learning baseline
against the physics features, which are told the shaft speed.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _block(c_in: int, c_out: int, k: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2)
    )


class BearingCNN(nn.Module):
    def __init__(self, n_classes: int = 4, width: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, width, 64, stride=8),
            _block(width, 2 * width, 3),
            _block(2 * width, 4 * width, 3),
            _block(4 * width, 4 * width, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _prep(signals: np.ndarray) -> torch.Tensor:
    x = np.asarray(signals, dtype=np.float32)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(x).unsqueeze(1)


class CNNClassifier:
    """sklearn-style wrapper: fit(signals, labels) / predict / predict_proba. Deterministic given `seed` on CPU."""

    def __init__(
        self, n_classes: int = 4, epochs: int = 25, batch_size: int = 64, lr: float = 2e-3, width: int = 16,
        seed: int = 0, threads: int = 4,
    ):
        self.n_classes, self.epochs, self.batch_size = n_classes, epochs, batch_size
        self.lr, self.width, self.seed, self.threads = lr, width, seed, threads
        self.model: BearingCNN | None = None
        self.history: list[float] = []

    def fit(self, signals: np.ndarray, labels: np.ndarray) -> CNNClassifier:
        labels = np.asarray(labels)
        if labels.min() < 0 or labels.max() >= self.n_classes:
            raise ValueError(f"labels must be in [0, {self.n_classes})")
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x, y = _prep(signals), torch.from_numpy(labels.astype(np.int64))
        counts = np.bincount(labels, minlength=self.n_classes).astype(np.float32)
        weights = torch.from_numpy(len(labels) / (self.n_classes * np.maximum(counts, 1)))  # balanced classes
        self.model = BearingCNN(self.n_classes, self.width)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * int(np.ceil(len(y) / self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.history = []
        for _ in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(y))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                idx = order[i : i + self.batch_size]
                xb = x[idx]
                shift = int(rng.integers(0, xb.shape[-1]))  # random circular time shift (augmentation)
                xb = torch.roll(xb, shifts=shift, dims=-1)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), y[idx])
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(idx)
            self.history.append(total / len(y))
        return self

    @torch.no_grad()
    def predict_proba(self, signals: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        self.model.eval()
        x = _prep(signals)
        out = [torch.softmax(self.model(x[i : i + 256]), dim=1) for i in range(0, len(x), 256)]
        return torch.cat(out).numpy()

    def predict(self, signals: np.ndarray) -> np.ndarray:
        return self.predict_proba(signals).argmax(axis=1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in (self.model or BearingCNN(self.n_classes, self.width)).parameters())
