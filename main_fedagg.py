"""Entry point for the FedAgg demo and first-stage algorithm upgrade."""

import argparse
import logging
import os
import random

import numpy as np
import torch

from data_loader import load_partition_data_cifar10
from fedagg import run_fedagg
from model_zoo import create_model


os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def add_args(parser):
    parser.add_argument("--data_dir", type=str, default="./data", help="data directory")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight decay parameter")
    parser.add_argument("--batch_size", type=int, default=8, help="input batch size for training")
    parser.add_argument("--comm_round", type=int, default=1000, help="maximum communication rounds")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--client_number", type=int, default=225, help="number of clients")
    parser.add_argument("--edge_number", type=int, default=15, help="number of edge nodes")
    parser.add_argument("--partition_method", type=str, default="hetero", help="data partition method")
    parser.add_argument("--partition_alpha", type=float, default=3.0, help="Dirichlet partition alpha")
    parser.add_argument("--method", type=str, default="fedagg", help="method name")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset used for training")
    parser.add_argument("--class_num", type=int, default=10, help="number of classes")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=0, help="numpy/data partition seed")
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=None,
        help="torch seed; default keeps the original demo behavior",
    )

    parser.add_argument("--T_agg", type=float, default=3.0, help="KD temperature")
    parser.add_argument(
        "--kd_weight_mode",
        type=str,
        default="adaptive",
        choices=["fixed", "adaptive"],
        help="use fixed or adaptive KD weights",
    )
    parser.add_argument("--kd_alpha_non_leaf", type=float, default=10.0, help="base KD weight for edge/cloud")
    parser.add_argument("--kd_alpha_leaf", type=float, default=1.0, help="base KD weight for clients")
    parser.add_argument("--leaf_ce_weight", type=float, default=1.0, help="local CE weight for leaf clients")
    parser.add_argument("--kd_warmup_rounds", type=int, default=10, help="rounds used to warm up KD strength")
    parser.add_argument("--kd_min_scale", type=float, default=0.2, help="minimum warmup scale for KD")
    parser.add_argument(
        "--kd_reliability_floor",
        type=float,
        default=0.25,
        help="minimum reliability multiplier for adaptive KD",
    )

    parser.add_argument(
        "--logit_topk",
        type=int,
        default=0,
        help="keep top-k delta logits during KD; 0 disables sparsification",
    )
    parser.add_argument(
        "--logit_quant_bits",
        type=int,
        default=0,
        help="quantize transmitted delta logits to this many bits; 0 keeps float values",
    )
    parser.add_argument("--track_comm", action="store_true", help="print simulated logit communication stats")

    args = parser.parse_args()
    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    args.client_number_per_round = args.client_number
    args.personal_learning_rate = args.lr
    args.client_num_in_total = args.client_number
    return args


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    else:
        raise Exception("dataset not implemented error")

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        test_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num_train,
        class_num_test,
    ) = data_loader(
        args.dataset,
        args.data_dir,
        args.partition_method,
        args.partition_alpha,
        args.client_number,
        args.batch_size,
    )

    return [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        test_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num_train,
        class_num_test,
    ]


def create_edge_model(args, n_classes, index):
    return create_model("resnet10")


def create_client_model(args, n_classes, index):
    return create_model("cnn")


def create_client_models(args, n_classes):
    random.seed(123)
    return [create_client_model(args, n_classes, index) for index in range(args.client_number)]


def create_edge_models(args, n_classes):
    random.seed(456)
    return [create_edge_model(args, n_classes, index) for index in range(args.edge_number)]


def create_cloud_model(args, n_classes):
    return create_model("resnet18")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    np.random.seed(args.seed)
    torch_seed = np.random.randint(5) if args.torch_seed is None else args.torch_seed
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    print("Seed/data_seed", args.seed, "torch_seed", torch_seed)

    dataset = load_data(args, args.dataset)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        test_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num_train,
        class_num_test,
    ] = dataset

    if args.method == "fedagg":
        client_models = create_client_models(args, class_num_train)
        edge_models = create_edge_models(args, class_num_train)
        cloud_model = create_cloud_model(args, class_num_train)
        run_fedagg(
            client_models,
            edge_models,
            cloud_model,
            train_data_local_num_dict,
            test_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            test_data_global,
            args,
        )
    else:
        raise Exception("method not implemented error")
