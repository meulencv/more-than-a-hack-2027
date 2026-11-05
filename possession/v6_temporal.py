"""Experimental temporal segmentation families; all targets come from other games.

Ideas adapted conceptually from temporal action segmentation (MS-TCN/ASRF), not
copies of their implementations. Boundary and segment losses use training labels
only. Local LayerNorm avoids the old model's sequence-length-dependent GroupNorm.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .sequence_tcn import Normalizer


class ChannelNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Residual(nn.Module):
    def __init__(self, width, dilation, local):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = ChannelNorm(width) if local else nn.GroupNorm(1, width)
        self.proj = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return x + self.proj(F.dropout(F.silu(self.norm(self.conv(x))), .1, self.training))


class SegmentNet(nn.Module):
    def __init__(self, features, width=64, local=True, multistage=False):
        super().__init__()
        self.input = nn.Conv1d(features, width, 1)
        self.blocks = nn.Sequential(*(Residual(width, 2 ** i, local) for i in range(7)))
        self.classifier = nn.Conv1d(width, 3, 1)
        self.boundary = nn.Conv1d(width, 1, 1)
        self.refiner = (nn.Sequential(nn.Conv1d(3, 32, 1),
                       *(Residual(32, 2 ** i, True) for i in range(5)), nn.Conv1d(32, 3, 1))
                       if multistage else None)

    def forward(self, x):
        h = self.blocks(self.input(x))
        first = self.classifier(h)
        final = first if self.refiner is None else first + self.refiner(first.softmax(1))
        return final, self.boundary(h)[:, 0], first


def segment_weights(y):
    y = np.asarray(y)
    cuts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    lengths = np.diff(cuts)
    typical = max(1, float(np.median(lengths)))
    weights = np.repeat(np.clip(np.sqrt(typical / np.maximum(lengths, 1)), .25, 4), lengths)
    weights[y < 0] = 0
    return weights.astype('float32')


def boundary_targets(y):
    y = np.asarray(y)
    edge = np.r_[False, (y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)].astype('float32')
    target = edge.copy()
    for offset, value in ((1, .5), (2, .25)):
        target[offset:] = np.maximum(target[offset:], edge[:-offset] * value)
        target[:-offset] = np.maximum(target[:-offset], edge[offset:] * value)
    target[y < 0] = 0
    return target


def objective(outputs, labels, weight, segment, boundary, use_boundary=False):
    logits, predicted_edge, first = outputs
    valid = labels >= 0
    ce = F.cross_entropy(logits, labels, weight=weight, ignore_index=-1, reduction='none')
    ce = (ce * segment * valid).sum() / (segment * valid).sum().clamp_min(1)
    same = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] == labels[:, :-1])
    lp = logits.log_softmax(1)
    smooth = ((lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2).clamp(max=16).mean(1)
    result = ce + .15 * (smooth * same).sum() / same.sum().clamp_min(1)
    if first is not logits:
        result = result + .3 * F.cross_entropy(first, labels, weight=weight, ignore_index=-1)
    if use_boundary:
        bce = F.binary_cross_entropy_with_logits(predicted_edge, boundary, reduction='none',
                                                 pos_weight=predicted_edge.new_tensor(8.))
        result = result + .2 * (bce * valid).sum() / valid.sum().clamp_min(1)
    return result


def train(xs, ys, kind='local', steps=800, seed=2027, device='mps'):
    torch.manual_seed(seed)
    torch.set_num_threads(6)
    rng = np.random.default_rng(seed)
    norm = Normalizer().fit(xs)
    values = [norm.transform(x) for x in xs]
    net = SegmentNet(values[0].shape[1], local=kind != 'global', multistage=kind == 'multistage').to(device)
    xx = [torch.from_numpy(a.T.copy()).to(device) for a in values]
    yy = [torch.tensor(y, dtype=torch.long, device=device) for y in ys]
    ss = [torch.from_numpy(segment_weights(y) if kind in ('segment','boundary') else
                          np.ones(len(y), dtype='float32')).to(device) for y in ys]
    bb = [torch.from_numpy(boundary_targets(y)).to(device) for y in ys]
    count = np.bincount(np.concatenate([y[y >= 0] for y in ys]), minlength=3)
    weight = torch.tensor(np.sqrt(count.sum() / np.maximum(count, 1)), device=device, dtype=torch.float32)
    weight /= weight.mean()
    opt = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.001)
    changes = [np.flatnonzero(y[1:] != y[:-1]) + 1 for y in ys]
    window, batch = 384, 12
    net.train()
    for step in range(steps):
        chunks = []
        for _ in range(batch):
            i = int(rng.integers(len(xs)))
            if len(xs[i]) < window:
                raise ValueError('Training sequence shorter than window')
            if len(changes[i]) and rng.random() < .5:
                start = int(np.clip(rng.choice(changes[i]) - rng.integers(window), 0, len(xs[i]) - window))
            else:
                start = int(rng.integers(len(xs[i]) - window + 1))
            chunks.append((i, slice(start, start + window)))
        inputs = torch.stack([xx[i][:, s] for i, s in chunks])
        target = torch.stack([yy[i][s] for i, s in chunks])
        sw = torch.stack([ss[i][s] for i, s in chunks])
        boundaries = torch.stack([bb[i][s] for i, s in chunks])
        opt.zero_grad(set_to_none=True)
        loss = objective(net(inputs), target, weight, sw, boundaries, kind == 'boundary')
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(kind, step, float(loss.detach().cpu()), flush=True)
    return net.eval(), norm


def predict(net, norm, arrays, device='mps'):
    probs, edges = [], []
    net.eval()
    with torch.inference_mode():
        for x in arrays:
            tensor = torch.from_numpy(norm.transform(x).T.copy()).to(device)[None]
            logits, boundary, _ = net(tensor)
            probs.append(logits.softmax(1)[0].T.cpu().numpy())
            edges.append(boundary.sigmoid()[0].cpu().numpy())
    return probs, edges
