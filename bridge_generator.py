"""Prototype-guided bridge carrier for FedAgg stage 3."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype import prototype_alignment_loss


class PrototypeBridgeGenerator(nn.Module):
    """Generate CIFAR-sized bridge samples conditioned on class prototypes."""

    def __init__(self, class_num=10, proto_dim=64, z_dim=64, hidden_dim=128):
        super().__init__()
        self.class_num = int(class_num)
        self.proto_dim = int(proto_dim)
        self.z_dim = int(z_dim)
        self.label_embed = nn.Embedding(self.class_num, self.proto_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.proto_dim * 2 + self.z_dim, hidden_dim * 4 * 4),
            nn.LayerNorm(hidden_dim * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 4, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, prototypes, labels, z=None):
        labels = labels.long().view(-1).clamp(0, self.class_num - 1)
        prototypes = prototypes.float()
        if z is None:
            z = torch.randn(prototypes.size(0), self.z_dim, device=prototypes.device)
        x = torch.cat([prototypes, self.label_embed(labels), z.float()], dim=1)
        x = self.fc(x).view(prototypes.size(0), -1, 4, 4)
        return self.decoder(x)


def lookup_condition_prototypes(bank, labels, proto_dim, device):
    """Look up label prototypes; missing classes get zero vectors."""
    labels = labels.long().view(-1).to(device)
    fallback = torch.zeros(labels.size(0), proto_dim, device=device)
    if bank is None:
        return fallback, torch.zeros(labels.size(0), dtype=torch.bool, device=device)
    prototypes, valid = bank.lookup(labels)
    prototypes = prototypes.to(device)
    valid = valid.to(device)
    return torch.where(valid.unsqueeze(1), prototypes, fallback), valid


def bridge_diversity_loss(images):
    """Small negative diversity term; lower loss means less batch collapse."""
    if images.size(0) <= 1:
        return images.sum() * 0.0
    flat = images.flatten(1)
    return -flat.std(dim=0, unbiased=False).mean()


def train_bridge_generator_step(node, teacher_model, labels, args):
    """Train a node-owned generator against the frozen teacher model."""
    if node.bridge_generator is None or node.bridge_optimizer is None or node.prototype_bank is None:
        return None

    train_steps = int(getattr(args, "bridge_train_steps", 1))
    if train_steps <= 0:
        return None

    labels = labels.to(node.device, dtype=torch.long)
    proto_dim = int(getattr(args, "proto_dim", 64))
    ce_weight = float(getattr(args, "bridge_ce_weight", 1.0))
    proto_weight = float(getattr(args, "bridge_proto_weight", 0.1))
    diversity_weight = float(getattr(args, "bridge_diversity_weight", 0.01))

    was_training = teacher_model.training
    teacher_model.eval()
    old_requires_grad = [param.requires_grad for param in teacher_model.parameters()]
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    last_loss = None
    node.bridge_generator.train()
    try:
        for _ in range(train_steps):
            condition_proto, _ = lookup_condition_prototypes(node.prototype_bank, labels, proto_dim, node.device)
            node.bridge_optimizer.zero_grad()
            generated = node.bridge_generator(condition_proto, labels)
            teacher_logits, teacher_features = teacher_model(generated)
            loss = ce_weight * F.cross_entropy(teacher_logits.float(), labels)
            if proto_weight > 0.0:
                loss = loss + proto_weight * prototype_alignment_loss(teacher_features, labels, node.prototype_bank)
            if diversity_weight > 0.0:
                loss = loss + diversity_weight * bridge_diversity_loss(generated)
            loss.backward()
            node.bridge_optimizer.step()
            last_loss = float(loss.detach().item())
    finally:
        for param, requires_grad in zip(teacher_model.parameters(), old_requires_grad):
            param.requires_grad_(requires_grad)
        if was_training:
            teacher_model.train()

    return last_loss


def sample_prototype_bridge(node, labels, args):
    """Sample generated bridge images from a node's prototype bank."""
    proto_dim = int(getattr(args, "proto_dim", 64))
    labels = labels.to(node.device, dtype=torch.long)
    condition_proto, valid = lookup_condition_prototypes(node.prototype_bank, labels, proto_dim, node.device)
    if node.bridge_generator is None or not valid.any():
        return None
    was_training = node.bridge_generator.training
    node.bridge_generator.eval()
    with torch.no_grad():
        generated = node.bridge_generator(condition_proto, labels)
    if was_training:
        node.bridge_generator.train()
    return generated
