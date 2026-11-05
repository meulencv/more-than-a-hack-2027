"""Small noncausal temporal CNN, independent of the HGB frame classifier.

Inspired by temporal action segmentation: classification plus bounded smoothness.
All normalization is fitted only on training matches. Never reads challenge labels.
"""
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Normalizer:
    def fit(self, arrays):
        x = np.concatenate(arrays)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.mean = np.nanmean(x, axis=0).astype('float32')
            self.std = np.nanstd(x, axis=0).astype('float32')
        self.mean = np.nan_to_num(self.mean)
        self.std = np.maximum(np.nan_to_num(self.std, nan=1), .1)
        return self

    def transform(self, x):
        missing = ~np.isfinite(x)
        z = (x - self.mean) / self.std
        z[missing] = 0
        return np.concatenate((np.clip(z, -8, 8), missing.astype('float32')), axis=1).astype('float32')


class Residual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation, padding=dilation)
        self.proj = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x):
        return x + self.proj(F.dropout(F.relu(self.norm(self.conv(x))), .1, self.training))


class TemporalCNN(nn.Module):
    def __init__(self, features, width=48, depth=6):
        super().__init__()
        self.input = nn.Conv1d(features, width, 1)
        self.blocks = nn.Sequential(*(Residual(width, 2 ** i) for i in range(depth)))
        self.out = nn.Conv1d(width, 3, 1)

    def forward(self, x):
        return self.out(self.blocks(self.input(x)))


def loss(logits, labels, weight=None, smooth=.15):
    classification = F.cross_entropy(logits, labels, weight=weight, ignore_index=-1)
    lp = F.log_softmax(logits, dim=1)
    temporal = torch.clamp((lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2, max=16).mean()
    return classification + smooth * temporal


def fit(sequences, labels, steps=500, seed=2027, device='cpu', width=48):
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    torch.set_num_threads(8)
    normalizer = Normalizer().fit(sequences)
    arrays = [normalizer.transform(x) for x in sequences]
    model = TemporalCNN(arrays[0].shape[1], width=width).to(device)
    train_x = [torch.from_numpy(x.T.copy()).to(device) for x in arrays]
    train_y = [torch.from_numpy(y.astype('int64')).to(device) for y in labels]
    all_labels = np.concatenate(labels)
    counts = np.bincount(all_labels[all_labels >= 0], minlength=3)
    weight = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1)), dtype=torch.float32, device=device)
    weight /= weight.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    model.train()
    batch, window = 16, 256
    boundaries = [np.flatnonzero(y[1:] != y[:-1]) + 1 for y in labels]
    for step in range(steps):
        xx, yy = [], []
        for _ in range(batch):
            i = int(np_rng.integers(len(arrays)))
            if np_rng.random() < .5 and len(boundaries[i]):
                center = int(np_rng.choice(boundaries[i]))
                start = int(np.clip(center - int(np_rng.integers(window)), 0, len(arrays[i]) - window))
            else:
                start = int(np_rng.integers(max(1, len(arrays[i]) - window)))
            xx.append(train_x[i][:, start:start + window])
            yy.append(train_y[i][start:start + window])
        optimizer.zero_grad(set_to_none=True)
        logits = model(torch.stack(xx))
        objective = loss(logits, torch.stack(yy), weight)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            print('step', step, 'loss', float(objective.detach().cpu()), flush=True)
    model.eval()
    return model, normalizer


def predict(model, normalizer, arrays, device='cpu'):
    out = []
    model.eval()
    with torch.inference_mode():
        for x in arrays:
            tensor = torch.from_numpy(normalizer.transform(x).T.copy()).to(device)[None]
            out.append(model(tensor).softmax(dim=1)[0].T.cpu().numpy())
    return out
