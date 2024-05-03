# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University
# all util functions for training.

import random
import numpy as np
import torch
import torch.nn as nn
from fairmotion.ops import conversions
from torch.nn import functional as F


def loss_constr_multi(ra, rb):
    # ra, rb in size (bs, 4*N)
    # rb is the model prediction, ra is GT
    assert ra.size() == rb.size()
    'check if size is valid, has to be a multiple of 4'
    assert (ra.size()[1] // 4) * 4 == ra.size()[1]

    'Again, you check the ground truth whether contains nan'
    mask = ~torch.any(ra.isnan(), dim=1)
    ra_c = ra[mask].clone()
    rb_c = rb[mask].clone()
    n_c = ra.size()[1] // 4

    loss_total = 0.0
    for i in range(n_c):
        start = 4*i
        'So here we are output a probability, which is filtered by the sigmoid function'
        c_l = F.binary_cross_entropy(torch.sigmoid(rb_c[:, start:start+1]), ra_c[:, start:start+1])
        # constr values might be too small numerically, *5.0?
        'Just the l2 mean'
        r_l = ((rb_c[:, start+1:start+4] - ra_c[:, start+1:start+4] * 5.0) ** 2).mean()
        loss_total += (c_l + r_l * 4.0)

    loss_total = loss_total / n_c * 2.5

    return loss_total


def loss_jerk(rb):
    # rb in size (bs, t, 18*6)
    # rb is the model prediction

    rb_c = rb.clone()  # maybe not necessary
    assert rb.size()[-1] == 18*6

    jitter = rb_c[:, 3:, :] - 3 * rb_c[:, 2:-1, :] + 3 * rb_c[:, 1:-2, :] - rb_c[:, :-3, :]

    return (jitter ** 2).mean() * 100.0


def loss_q_only_2axis(ra, rb):
    # ra, rb in size (bs, 18*6 +3)
    # rb is the model prediction, ra is GT

    assert ra.size() == rb.size()
    ra_c, rb_c = ra.clone(), rb.clone()  # maybe not necessary
    assert ra.size()[1] == 18*6+3

    'Here we focus on the rotation matrix'
    r2_a = ra_c[:, :-3]
    r2_b = rb_c[:, :-3]
    loss_q = ((r2_b - r2_a) ** 2).mean() * 100.0

    xy_a = ra_c[:, -3:-1]
    xy_b = rb_c[:, -3:-1]

    'Notice that the nan check is for ground truth (a). So ground truth is problematic'
    mask = ~torch.any(xy_a.isnan(), dim=1)
    xy_a = xy_a[mask]
    xy_b = xy_b[mask]
    'focus on the xy,and avoid the nan case, with weights 6.0'
    loss_dq_root1 = ((xy_a - xy_b) ** 2).mean() * 6.0

    z_a = ra_c[:, -1:]
    z_b = rb_c[:, -1:]

    z_a = z_a[mask]
    z_b = z_b[mask]

    'focus on z, give it a little bit higher weights, 12.0'
    loss_dq_root2 = ((z_a - z_b) ** 2).mean() * 12.0

    return loss_q + loss_dq_root1 + loss_dq_root2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out
