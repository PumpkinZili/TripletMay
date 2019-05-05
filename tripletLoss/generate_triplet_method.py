import torch
import torch.nn
import numpy as np
import random


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        # dist = dist - torch.diag(dist.diag)
        dist = dist - torch.diag(dist)

    return torch.clamp(dist, 0.0, np.inf)


def generate_k_triplet(embeddeds, targets, K=2, B=2):
    """
    choose close k as the positive and close B for negative
    generate N * K * B triplets
    work in cuda
    """
    batch_len = embeddeds.size(0)

    dis_mat = pairwise_distances(embeddeds).cpu().data.numpy()

    anchor, positive, negative = [], [], []

    ts = targets.reshape(-1).cpu().data.numpy()

    for i in range(batch_len):
        incls_id = np.where(ts == ts[i])[0]
        incls = dis_mat[i][incls_id]

        outcls_id = np.where(ts != ts[i])[0]
        outcls = dis_mat[i][outcls_id]

        incls_closeK = np.argsort(incls)[1:1 + K]
        outcls_closeB = np.argsort(outcls)[0:B]

        if len(incls_closeK) == 0 or len(outcls_closeB) == 0:
            continue

        an = embeddeds[i].unsqueeze(0)
        for c in incls_closeK:
            for o in outcls_closeB:
                anchor.append(an)
                positive.append(embeddeds[incls_id[c]].unsqueeze(0))
                negative.append(embeddeds[outcls_id[o]].unsqueeze(0))
    try:
        anchor = torch.cat(anchor, 0)
        positive = torch.cat(positive, 0)
        negative = torch.cat(negative, 0)
    except RuntimeError:
        print(dis_mat)
        print(anchor)
        print(positive)
        print(negative)
        print(targets)
    return anchor, positive, negative


def generate_random_triplets(embeddeds, targets, triplet_num):
    """
    generate random triplets
    :param embeddeds:
    :param targets:
    :param triplet_num: number of triplets to generate
    :return:
    """
    # print("generate random triplet")
    batch_len = embeddeds.size(0)

    ts = targets.reshape(-1).cpu().data.numpy()
    anchor, positive, negative = [], [], []
    for i in range(triplet_num):
        an_id = random.randint(0, batch_len - 1)
        incls_ids = np.where(ts == ts[an_id])[0]
        while len(incls_ids) == 1:
            an_id = random.randint(0, batch_len - 1)
            incls_ids = np.where(ts == ts[an_id])[0]

        pos_id = random.choice(incls_ids)
        while pos_id == an_id:
            pos_id = random.choice(incls_ids)

        outcls_ids = np.where(ts != ts[an_id])[0]
        neg_id = random.choice(outcls_ids)

        anchor.append(embeddeds[an_id].unsqueeze(0))
        positive.append(embeddeds[pos_id].unsqueeze(0))
        negative.append(embeddeds[neg_id].unsqueeze(0))

    anchor = torch.cat(anchor, 0)
    positive = torch.cat(positive, 0)
    negative = torch.cat(negative, 0)

    return anchor, positive, negative


def generate_batch_hard_triplet(embeddeds, targets):
    """Batch Hard
    Args:
        embeddeds
        targets
    Returns:
        anchor, positive, negative
    """
    batch_len = embeddeds.size(0)

    dis_mat = pairwise_distances(embeddeds).cpu().data.numpy()
    anchor, positive, negative = [], [], []

    ts = targets.reshape(-1).cpu().data.numpy()

    for i in range(batch_len):
        incls_id = np.nonzero(ts == ts[i])[0]  # numpy
        incls = dis_mat[i][incls_id]

        outcls_id = np.nonzero(ts != ts[i])[0]  # nunpy
        outcls = dis_mat[i][outcls_id]

        if incls_id.size <= 1 or outcls_id.size < 1:
            continue

        incls_farest = np.argsort(incls)[-1]
        outcls_closest = np.argsort(outcls)[0]

        an = embeddeds[i].unsqueeze(0)
        anchor.append(an)
        positive.append(embeddeds[incls_farest].unsqueeze(0))
        negative.append(embeddeds[outcls_closest].unsqueeze(0))

    try:
        anchor = torch.cat(anchor, 0)
        positive = torch.cat(positive, 0)
        negative = torch.cat(negative, 0)
    except RuntimeError:
        print(anchor)
        print(positive)
        print(negative)

    return anchor, positive, negative


def generate_all_triplet(embeddeds, targets):
    batch_len = embeddeds.size(0)
    ts = targets.reshape(-1).cpu().data.numpy()

    un_embeddeds = embeddeds.unsqueeze(dim=1)

    anchor, positive, negative = [], [], []

    for i in range(batch_len):
        incls_id = np.nonzero(ts == ts[i])[0]

        outcls_id = np.nonzero(ts != ts[i])[0]

        if incls_id.size <= 1 or outcls_id.size < 1:
            continue

        for iid in incls_id:
            if iid == i:
                continue
            for oid in outcls_id:
                anchor.append(un_embeddeds[i])
                positive.append(un_embeddeds[iid])
                negative.append(un_embeddeds[oid])
    try:
        anchor = torch.cat(anchor, 0)
        positive = torch.cat(positive, 0)
        negative = torch.cat(negative, 0)
    except Exception as e:
        print(anchor)
        print(positive)
        print(negative)
        raise RuntimeError

    return anchor, positive, negative

# def generate_triplet(embeddeds, labels, K=2):
#    """
#    choose close k as the positive, and closest as the negative
#    Args:
#        embeddeds:
#        labels:
#        K: close K
#    """
#
#    batch_len = embeddeds.size(0)
#    distance = embeddeds
#
#    dis_mat = np.zeros((batch_len, batch_len))
#    for i, e1 in enumerate(embeddeds):
#        for j, e2 in enumerate(embeddeds):
#            if i != j:
#                dis = (e1 - e2).norm(2)
#                dis_mat[i, j] = dis_mat[j, i] = dis
#
#    anchor, positive, negative = [], [], []
#
#    l = labels.reshape(-1).cpu().data.numpy()
#    for i in range(batch_len):
#        incls_id = np.where(l == l[i])[0]
#        incls = dis_mat[i][incls_id]
#        # print(incls_id)
#        outcls_id = np.where(l != l[i])[0]
#        outcls = dis_mat[i][outcls_id]
#        # print(outcls_id)
#
#        closeK = np.argsort(incls)[1:1 + K]
#        # print(closeK)
#        out1 = np.argmin(outcls)
#        # print(out1)
#
#        an = embeddeds[i].unsqueeze(0)
#        out = embeddeds[outcls_id[out1]].unsqueeze(0)
#        for c in closeK:
#            anchor.append(an)
#            positive.append(embeddeds[incls_id[c]].unsqueeze(0))
#            negative.append(out)
#
#    anchor = torch.cat(anchor, 0)
#    positive = torch.cat(positive, 0)
#    negative = torch.cat(negative, 0)
#    return anchor, positive, negative
