import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelArgs:
    patch_size: int = 8
    batch_size: int = 32
    image_size: int = 64
    assert image_size % patch_size == 0
    groups = image_size//patch_size
    patch_num = groups * groups
    max_seq_len = patch_num + 1
    dim = 384
    in_channels = 3
    base = 10000
    n_q_head = 6
    n_encoder = 6
    n_kv_head = 3
    assert n_q_head % n_kv_head == 0
    n_kv_group = n_q_head // n_kv_head
    assert dim % n_q_head == 0
    head_dim = dim//n_q_head
    num_classes = 200
    dropout = 0.1

    eps = 1e-4

    mode = 'train'
    lr = 1e-5
    num_epochs: int = 200
    model_folder: str = '/scratch/vdudekul/checkpoints/vit_tinyimagenet_train2'
    data_folder_path: str = '/scratch/vdudekul/checkpoints/datasets_collections/tinyimagenet'

