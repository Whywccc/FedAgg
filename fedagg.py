"""Core FedAgg training loop.

This version keeps the original end-edge-cloud bidirectional distillation
flow, and adds a first-stage algorithm upgrade:

1. fixed or adaptive KD weights;
2. optional top-k / quantized delta-logit simulation;
3. communication byte accounting;
4. safer device handling and independent child model assignment.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

import utils
from autoencoder_pretrained import create_autoencoder
from utils import KL_Loss


def get_device(args):
    return torch.device(getattr(args, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))


def get_temperature(args):
    temperature = getattr(args, "T_agg", 3.0)
    if isinstance(temperature, (list, tuple)):
        temperature = temperature[0]
    return float(temperature)


def _get_arg(args, name, default):
    return getattr(args, name, default)


def compute_adaptive_kd_weight(student_logits, teacher_logits, args, base_alpha):
    """Use teacher confidence, teacher-student agreement, and round warmup."""
    if _get_arg(args, "kd_weight_mode", "fixed") != "adaptive":
        return float(base_alpha)

    with torch.no_grad():
        teacher_prob = F.softmax(teacher_logits, dim=1)
        student_prob = F.softmax(student_logits, dim=1)
        class_num = teacher_prob.size(1)

        entropy = -(teacher_prob * torch.log(teacher_prob.clamp_min(1e-8))).sum(dim=1)
        confidence = 1.0 - entropy / math.log(class_num)
        confidence = confidence.clamp(0.0, 1.0).mean()

        agreement = F.cosine_similarity(student_prob, teacher_prob, dim=1)
        agreement = ((agreement + 1.0) * 0.5).clamp(0.0, 1.0).mean()
        reliability = 0.5 * confidence + 0.5 * agreement
        reliability = reliability.clamp(float(_get_arg(args, "kd_reliability_floor", 0.25)), 1.0)

        warmup_rounds = max(int(_get_arg(args, "kd_warmup_rounds", 10)), 1)
        current_round = int(_get_arg(args, "current_round", 0))
        warmup = min(1.0, float(current_round + 1) / float(warmup_rounds))
        min_scale = float(_get_arg(args, "kd_min_scale", 0.2))
        round_scale = min_scale + (1.0 - min_scale) * warmup

        return max(float(base_alpha) * round_scale * float(reliability.item()), 0.0)


def _uniform_quantize(values, bits):
    if bits <= 0 or values.numel() == 0:
        return values

    levels = float((1 << bits) - 1)
    min_val = values.min()
    max_val = values.max()
    span = (max_val - min_val).clamp_min(1e-8)
    quantized = torch.round((values - min_val) / span * levels)
    return quantized / levels * span + min_val


def _add_comm_stats(args, dense_bytes, compressed_bytes):
    args.comm_dense_bytes = _get_arg(args, "comm_dense_bytes", 0.0) + float(dense_bytes)
    args.comm_compressed_bytes = _get_arg(args, "comm_compressed_bytes", 0.0) + float(compressed_bytes)


def prepare_teacher_logits_for_kd(teacher_logits, args):
    """Simulate transmitting mean + top-k quantized delta logits.

    The reconstructed logits are used for KD, so enabling compression affects
    both communication accounting and the actual distillation signal.
    """
    topk = int(_get_arg(args, "logit_topk", 0))
    bits = int(_get_arg(args, "logit_quant_bits", 0))
    dense_bytes = teacher_logits.numel() * teacher_logits.element_size()

    if topk <= 0 and bits <= 0:
        _add_comm_stats(args, dense_bytes, dense_bytes)
        return teacher_logits

    class_num = teacher_logits.size(1)
    topk = class_num if topk <= 0 else min(topk, class_num)
    baseline = teacher_logits.mean(dim=1, keepdim=True)
    delta = teacher_logits - baseline
    _, indices = torch.topk(delta.abs(), k=topk, dim=1)
    selected = delta.gather(1, indices)
    selected = _uniform_quantize(selected, bits)

    reconstructed_delta = torch.zeros_like(delta)
    reconstructed_delta.scatter_(1, indices, selected)
    reconstructed = baseline + reconstructed_delta

    index_bytes = indices.numel() * 2
    if bits > 0:
        value_bytes = selected.numel() * bits / 8.0
        scale_bytes = teacher_logits.size(0) * 8.0
    else:
        value_bytes = selected.numel() * teacher_logits.element_size()
        scale_bytes = 0.0
    baseline_bytes = baseline.numel() * teacher_logits.element_size()
    _add_comm_stats(args, dense_bytes, baseline_bytes + index_bytes + value_bytes + scale_bytes)
    return reconstructed


def maybe_print_comm_stats(args, comm_round):
    if not _get_arg(args, "track_comm", False):
        return

    dense = float(_get_arg(args, "comm_dense_bytes", 0.0))
    compressed = float(_get_arg(args, "comm_compressed_bytes", 0.0))
    if dense <= 0:
        return

    print(
        "Comm/Logits in comm_round",
        comm_round,
        "dense_MB",
        round(dense / (1024 * 1024), 4),
        "effective_MB",
        round(compressed / (1024 * 1024), 4),
        "ratio",
        round(compressed / dense, 4),
    )


def test_on_cloud(cloud_model, test_data_global, comm_round):
    device = next(cloud_model.parameters()).device
    was_training = cloud_model.training
    cloud_model.eval()

    accTop1_avg = utils.RunningAverage()
    accTop5_avg = utils.RunningAverage()
    with torch.no_grad():
        for images, labels in test_data_global:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long)
            log_probs, _ = cloud_model(images)
            metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())

    print("Test/AccTop1 in comm_round", comm_round, accTop1_avg.value())
    if was_training:
        cloud_model.train()


def run_fedagg(
    client_models,
    edge_models,
    cloud_model,
    train_data_local_num_dict,
    test_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    test_data_global,
    args,
):
    V1 = [Node(args, cloud_model)]
    V2 = create_child_for_upper_level(args, V1, args.edge_number, edge_models)
    assert args.client_number % args.edge_number == 0
    V3 = create_child_for_upper_level(
        args,
        V2,
        args.client_number // args.edge_number,
        client_models,
    )

    for idx, node in enumerate(V3):
        node.dataset = [_ for _ in train_data_local_dict[idx]]

    Init(V1[0])
    args.comm_dense_bytes = 0.0
    args.comm_compressed_bytes = 0.0

    for comm_round in range(args.comm_round):
        args.current_round = comm_round
        train_FedAgg(V1[0], args)
        test_on_cloud(V1[0].model, test_data_global, comm_round)
        maybe_print_comm_stats(args, comm_round)


global_index = 0


class Node:
    def __init__(self, args, model, father=None):
        global global_index

        self.device = get_device(args)
        self.model = model.to(self.device)
        self.dataset = None
        self.father = father
        self.children = []
        self.index = global_index
        self.autoencoder = create_autoencoder(self.device)
        self.noises = []
        self.labels = []
        self.args = args
        global_index += 1

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.father is None

    def __repr__(self):
        father = "None" if self.father is None else str(self.father.index)
        children = [child.index for child in self.children]
        return f"[index:{self.index} father:{father} children:{children}]"


def create_child_for_upper_level(args, upper_level, children_number, models):
    result = []
    model_offset = 0
    for parent in upper_level:
        child_models = models[model_offset : model_offset + children_number]
        if len(child_models) != children_number:
            raise ValueError("Not enough model instances for the requested FedAgg tree.")
        model_offset += children_number

        sub_nodes = [Node(args, model, parent) for model in child_models]
        parent.children = sub_nodes
        result.extend(sub_nodes)
    return result


def train_FedAgg(node, args):
    if node.is_root():
        for child in node.children:
            train_FedAgg(child, args)
    elif node.is_leaf():
        BSBODP(node, node.father, args)
    else:
        for child in node.children:
            train_FedAgg(child, args)
        BSBODP(node, node.father, args)


def Init(node):
    if node.is_root():
        for child in node.children:
            Init(child)
    elif node.is_leaf():
        for img, label in node.dataset:
            img = img.to(node.device)
            label = label.to(device=node.device, dtype=torch.long)
            with torch.no_grad():
                noise = node.autoencoder.encoder(img)
            node.noises.append(noise.detach())
            node.labels.append(label.detach())
        node.father.noises.extend(node.noises)
        node.father.labels.extend(node.labels)
    else:
        for child in node.children:
            Init(child)
        node.father.noises.extend(node.noises)
        node.father.labels.extend(node.labels)


def BSBODP(node1, node2, args):
    BSBODP_dir(node1, node2, args)
    BSBODP_dir(node2, node1, args)


class Loss_Non_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=10):
        super(Loss_Non_Leaf, self).__init__()
        self.alpha = alpha
        self.kl_loss_crit = KL_Loss(temperature)
        self.ce_loss_crit = nn.CrossEntropyLoss()

    def forward(self, output_batch, teacher_outputs, label, kd_weight=None):
        loss_ce = self.ce_loss_crit(output_batch, label.long())
        loss_kl = self.kl_loss_crit(output_batch, teacher_outputs.detach())
        alpha = self.alpha if kd_weight is None else kd_weight
        return loss_ce + alpha * loss_kl


class Loss_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=1, alpha2=1):
        super(Loss_Leaf, self).__init__()
        self.alpha = alpha
        self.alpha2 = alpha2
        self.non_leaf_loss_crit = Loss_Non_Leaf(temperature, alpha)
        self.ce_loss_crit = nn.CrossEntropyLoss()

    def forward(self, output_fake, teacher_outputs_fake, output_true, label, kd_weight=None):
        loss_leaf = self.non_leaf_loss_crit(
            output_fake,
            teacher_outputs_fake.detach(),
            label.long(),
            kd_weight=kd_weight,
        )
        loss_ce = self.ce_loss_crit(output_true, label.long())
        return loss_leaf + self.alpha2 * loss_ce


def BSBODP_dir(node_origin, node_neigh, args):
    if len(node_neigh.noises) < len(node_origin.noises):
        noises = node_neigh.noises
        labels = node_neigh.labels
    else:
        noises = node_origin.noises
        labels = node_origin.labels

    device = node_origin.device
    temperature = get_temperature(args)
    non_leaf_alpha = float(_get_arg(args, "kd_alpha_non_leaf", 10.0))
    leaf_alpha = float(_get_arg(args, "kd_alpha_leaf", 1.0))
    leaf_ce_weight = float(_get_arg(args, "leaf_ce_weight", 1.0))

    crit_non_leaf = Loss_Non_Leaf(temperature, non_leaf_alpha).to(device)
    crit_leaf = Loss_Leaf(temperature, leaf_alpha, leaf_ce_weight).to(device)
    optimizer = torch.optim.SGD(node_origin.model.parameters(), lr=node_origin.args.lr, momentum=0.9)

    node_origin.model.train()
    node_neigh.model.eval()
    node_neigh.autoencoder.eval()

    for idx, (noise, label) in enumerate(zip(noises, labels)):
        optimizer.zero_grad()
        noise = noise.to(device)
        label = label.to(device=device, dtype=torch.long)

        with torch.no_grad():
            fake_data = node_neigh.autoencoder.decoder(noise)
            teacher_logits, _ = node_neigh.model(fake_data)
            teacher_logits = prepare_teacher_logits_for_kd(teacher_logits, args)

        student_logits, _ = node_origin.model(fake_data)

        if node_origin.is_leaf():
            img, local_label = node_origin.dataset[idx]
            img = img.to(device)
            local_label = local_label.to(device=device, dtype=torch.long)
            if not torch.equal(label.detach().cpu(), local_label.detach().cpu()):
                raise AssertionError("FedAgg latent label and local label are not aligned.")
            true_logits, _ = node_origin.model(img)
            kd_weight = compute_adaptive_kd_weight(student_logits, teacher_logits, args, leaf_alpha)
            loss = crit_leaf(student_logits, teacher_logits, true_logits, label, kd_weight=kd_weight)
        else:
            kd_weight = compute_adaptive_kd_weight(student_logits, teacher_logits, args, non_leaf_alpha)
            loss = crit_non_leaf(student_logits, teacher_logits, label, kd_weight=kd_weight)

        loss.backward()
        optimizer.step()
