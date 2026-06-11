"""Prototype bank utilities for FedAgg stage 2.

The bank stores one normalized feature prototype per class. It is deliberately
small and independent from the main training loop so it can be enabled as a
side path first, then used by a prototype loss when the experiment is stable.
"""

import torch
import torch.nn.functional as F


def feature_to_vector(features, proto_dim):
    """Pool model features to normalized vectors with a fixed dimension."""
    features = features.float()
    if features.dim() > 2:
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
    elif features.dim() == 1:
        features = features.unsqueeze(0)

    feature_dim = features.size(1)
    if feature_dim > proto_dim:
        features = features[:, :proto_dim]
    elif feature_dim < proto_dim:
        pad = features.new_zeros(features.size(0), proto_dim - feature_dim)
        features = torch.cat([features, pad], dim=1)

    return F.normalize(features, p=2, dim=1, eps=1e-8)


def batch_prototype_sums(features, labels, class_num, proto_dim):
    """Return per-class feature sums and sample counts for one batch."""
    vectors = feature_to_vector(features, proto_dim).detach()
    labels = labels.detach().long().view(-1).to(vectors.device)
    valid = (labels >= 0) & (labels < class_num)

    sums = vectors.new_zeros(class_num, proto_dim)
    counts = vectors.new_zeros(class_num)
    if valid.any():
        valid_labels = labels[valid]
        sums.index_add_(0, valid_labels, vectors[valid])
        counts.index_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=vectors.dtype))
    return sums, counts


class PrototypeBank:
    """EMA-updated class prototype table."""

    def __init__(self, class_num, proto_dim, device):
        self.class_num = int(class_num)
        self.proto_dim = int(proto_dim)
        self.prototypes = torch.zeros(self.class_num, self.proto_dim, device=device)
        self.counts = torch.zeros(self.class_num, device=device)

    def update(self, means, counts, momentum):
        means = means.to(self.prototypes.device).float()
        counts = counts.to(self.counts.device).float()
        mask = counts > 0
        if not mask.any():
            return

        momentum = float(momentum)
        existing = self.counts > 0
        replace_mask = mask & ~existing
        ema_mask = mask & existing

        self.prototypes[replace_mask] = means[replace_mask]
        self.prototypes[ema_mask] = (
            momentum * self.prototypes[ema_mask] + (1.0 - momentum) * means[ema_mask]
        )
        self.prototypes[mask] = F.normalize(self.prototypes[mask], p=2, dim=1, eps=1e-8)
        self.counts[mask] += counts[mask]

    def update_from_batch(self, features, labels, momentum):
        sums, counts = batch_prototype_sums(features, labels, self.class_num, self.proto_dim)
        means = sums / counts.clamp_min(1.0).unsqueeze(1)
        self.update(means, counts, momentum)

    def lookup(self, labels):
        labels = labels.detach().long().view(-1).to(self.prototypes.device)
        valid = (labels >= 0) & (labels < self.class_num) & (self.counts[labels.clamp(0, self.class_num - 1)] > 0)
        safe_labels = labels.clamp(0, self.class_num - 1)
        return self.prototypes[safe_labels].detach(), valid

    def coverage(self):
        return int((self.counts > 0).sum().item())

    def mean_norm(self):
        mask = self.counts > 0
        if not mask.any():
            return 0.0
        return float(self.prototypes[mask].norm(dim=1).mean().item())


def aggregate_prototype_banks(banks, class_num, proto_dim, device):
    """Aggregate child banks with sample-count weighted averaging."""
    sums = torch.zeros(class_num, proto_dim, device=device)
    counts = torch.zeros(class_num, device=device)
    for bank in banks:
        if bank is None:
            continue
        child_counts = bank.counts.to(device).float()
        mask = child_counts > 0
        if not mask.any():
            continue
        sums[mask] += bank.prototypes.to(device)[mask] * child_counts[mask].unsqueeze(1)
        counts[mask] += child_counts[mask]

    means = sums / counts.clamp_min(1.0).unsqueeze(1)
    means[counts > 0] = F.normalize(means[counts > 0], p=2, dim=1, eps=1e-8)
    return means, counts


def prototype_alignment_loss(features, labels, bank):
    """Cosine alignment loss against the target class prototypes."""
    if bank is None:
        return features.float().sum() * 0.0

    vectors = feature_to_vector(features, bank.proto_dim)
    targets, valid = bank.lookup(labels)
    valid = valid.to(vectors.device)
    if not valid.any():
        return vectors.sum() * 0.0

    targets = targets.to(vectors.device)
    return (1.0 - F.cosine_similarity(vectors[valid], targets[valid], dim=1)).mean()
