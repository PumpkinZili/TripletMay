"""train with kTriplet, BatchHardTriplet, BatchAlltriplet"""

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from losses import CenterLoss
from generate_triplet_method import generate_k_triplet, pairwise_distances, generate_batch_hard_triplet, \
    generate_all_triplet
from utils import get_lr, get_metrics


def fit(train_loader, sampler_train_loader, test_loader, test_fc_loader, model, loss_fn, optimzer, scheduler, n_epochs, k, n_K, log_interval,
        shuffle_interval=-1, global_loss=False, writer=None, start_epoch=0, n_classes=10, gamma=0.01, use_sampler=False, rm_zero=True,
        method='kTriplet', data_augmentation=False, center_sigma=-1.0, freeze_parameter=False, use_cross_entropy=False):

    # When freeze parameter, set n_epochs=0
    if freeze_parameter:
        epoch = 0
        train_embeddeds, train_targets = get_train_embeddeds(train_loader, model)

        train_predictK = [k]
        accs, mcas, mcps, crs, cps = predictKNN(train_embeddeds, train_targets, train_loader,
                                                model, n_classes, k=train_predictK,
                                                train=True, data_augmentation=False)

        # Add Train scalar
        writer.add_scalar('Train/accuracy', accs[k], epoch)
        writer.add_scalar('Train/mca', mcas[k], epoch)
        writer.add_scalar('Train/mcp', mcps[k], epoch)
        for i, recall in enumerate(crs[k]):
            writer.add_scalar('Train_recall/{}_recall'.format(i), recall, epoch)
        for i, precision in enumerate(cps[k]):
            writer.add_scalar('Train_precision/{}_recall'.format(i), precision, epoch)
        print(
            'Train set: Accuracy: {} MCA: {} MCP: {}'.format(accs[k], mcas[k], mcps[k]))

        # Test stage
        #test_loss = test_epoch(test_loader, model, loss_fn, k, n_K, rm_zero, n_classes)
        test_loss = 0
        test_predictK = [1, 3, 5, 7, 9]
        if data_augmentation == True:
            accs, mcas, mcps, crs, cps = predictKNN(train_embeddeds, train_targets, test_fc_loader,
                                                    model, n_classes, k=test_predictK,
                                                    train=False,
                                                    data_augmentation=data_augmentation)
        else:
            accs, mcas, mcps, crs, cps = predictKNN(train_embeddeds, train_targets, test_loader,
                                                    model, n_classes, k=test_predictK, train=False,
                                                    data_augmentation=data_augmentation)

        # Add Test scalar
        writer.add_scalar('Test/accuracy', accs[k], epoch)
        writer.add_scalar('Test/mca', mcas[k], epoch)
        writer.add_scalar('Test/mcp', mcps[k], epoch)
        for pk in test_predictK:
            writer.add_scalar('TestPredictK/accuracy{}'.format(pk), accs[pk], epoch)
            writer.add_scalar('TestPredictK/mca_{}'.format(pk), mcas[pk], epoch)
            writer.add_scalar('TestPredictK/mcp_{}'.format(pk), mcps[pk], epoch)
        for i, recall in enumerate(crs[k]):
            writer.add_scalar('Test_recall/{}_recall'.format(i), recall, epoch)
        for i, precision in enumerate(cps[k]):
            writer.add_scalar('Test_precision/{}_recall'.format(i), precision, epoch)

        print(
            'Test set: Average loss: {:.4f}\tAccuracy: {} MCA: {} MCP: {}\n'.format(test_loss, accs[k], mcas[k],
                                                                                        mcps[k]))
        n_epochs = 0


    for epoch in range(0, start_epoch):
        if scheduler is not None:
            scheduler.step()

    if global_loss and shuffle_interval == 0:
        raise RuntimeError("When global_loss=True, shuffle_interval should not be zero.")

    for epoch in range(start_epoch, n_epochs):
        print("Epoch {} / {}:".format(epoch + 1, n_epochs))

        if scheduler is not None:
            scheduler.step()

        # Shuffle
        if shuffle_interval > 0 and epoch % shuffle_interval == 0:
            print("Shuffle...")
            train_loader.dataset.shuffle()

        # draw learning rate on tensorboard
        writer.add_scalar('Train/Learning_Rate', get_lr(optimzer), epoch)

        if epoch == 0:
            if global_loss:
                train_embeddeds, train_targets = get_train_embeddeds(train_loader, model)
            else:
                train_embeddeds, train_targets = None, None

        # Train stage, average_inner_class dont't mutiply by gamma
        if use_sampler:
            train_loss, average_triplet_loss, average_inner_class, average_center_loss, average_cross_entropy = train_epoch(
                sampler_train_loader, model, loss_fn,
                optimzer, k, n_K,
                n_classes, rm_zero, global_loss,
                train_embeddeds, train_targets, gamma,
                center_sigma, method, use_cross_entropy)
        else:
            train_loss, average_triplet_loss, average_inner_class, average_center_loss, average_cross_entropy = train_epoch(train_loader,
                                                                                                     model, loss_fn,
                                                                                                     optimzer,
                                                                                                     k, n_K,
                                                                                                     n_classes, rm_zero,
                                                                                                     global_loss,
                                                                                                     train_embeddeds,
                                                                                                     train_targets,
                                                                                                     gamma,
                                                                                                     center_sigma,
                                                                                                     method,
                                                                                                     use_cross_entropy)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Triplet_loss', average_triplet_loss, epoch)
        writer.add_scalar('Train/Inner_class_loss', average_inner_class, epoch)
        writer.add_scalar('Train/Center_loss', average_center_loss, epoch)

        if (use_sampler and epoch % 25 == 0) or (not use_sampler):
            train_embeddeds, train_targets = get_train_embeddeds(train_loader, model)

            train_predictK = [k]
            accs, mcas, mcps, crs, cps = predictKNN(train_embeddeds, train_targets, train_loader,
                                                    model, n_classes, k=train_predictK,
                                                    train=True, data_augmentation=False)

            # Add Train scalar
            writer.add_scalar('Train/accuracy', accs[k], epoch)
            writer.add_scalar('Train/mca', mcas[k], epoch)
            writer.add_scalar('Train/mcp', mcps[k], epoch)
            for i, recall in enumerate(crs[k]):
                writer.add_scalar('Train_recall/{}_recall'.format(i), recall, epoch)
            for i, precision in enumerate(cps[k]):
                writer.add_scalar('Train_precision/{}_recall'.format(i), precision, epoch)
            print(
                'Train set: Average loss: {:.4f}\tAccuracy: {} MCA: {} MCP: {}'.format(train_loss, accs[k], mcas[k],
                                                                                       mcps[k]))

            # Test stage
            #test_loss = test_epoch(test_loader, model, loss_fn, k, n_K, rm_zero, n_classes)
            test_loss = 0
            test_predictK = [1, 3, 5, 7, 9]
            if data_augmentation == True:
                accs, mcas, mcps, crs, cps = predictKNN(train_embeddeds, train_targets, test_fc_loader,
                                                        model, n_classes, k=test_predictK,
                                                        train=False,
                                                        data_augmentation=data_augmentation)
            else:
                accs, mcas, mcps, crs, cps = predictKNN(train_embeddeds, train_targets, test_loader,
                                                        model, n_classes, k=test_predictK, train=False,
                                                        data_augmentation=data_augmentation)

            # Add Test scalar
            writer.add_scalar('Test/accuracy', accs[k], epoch)
            writer.add_scalar('Test/mca', mcas[k], epoch)
            writer.add_scalar('Test/mcp', mcps[k], epoch)
            for pk in test_predictK:
                writer.add_scalar('TestPredictK/accuracy{}'.format(pk), accs[pk], epoch)
                writer.add_scalar('TestPredictK/mca_{}'.format(pk), mcas[pk], epoch)
                writer.add_scalar('TestPredictK/mcp_{}'.format(pk), mcps[pk], epoch)
            for i, recall in enumerate(crs[k]):
                writer.add_scalar('Test_recall/{}_recall'.format(i), recall, epoch)
            for i, precision in enumerate(cps[k]):
                writer.add_scalar('Test_precision/{}_recall'.format(i), precision, epoch)

            print(
                'Test set: Average loss: {:.4f}\tAccuracy: {} MCA: {} MCP: {}\n'.format(test_loss, accs[k], mcas[k],
                                                                                        mcps[k]))

            writer.add_scalar('Test/Loss', test_loss, epoch)


def train_epoch(train_loader, model, loss_fn, optimzer, k, n_K, n_classes, rm_zero,
                global_loss=False, train_embeddeds=None, train_targets=None, gamma=0.01, center_sigma=-1.0,
                method='kTriplet', use_cross_entropy=False):
    model.train()
    losses_triplet = []
    losses_inner_class = []
    losses_center = []
    losses_cross_entropy = []
    losses = []
    center_loss = CenterLoss(n_classes, mean_center=True).cuda()
    cross_entropy = nn.CrossEntropyLoss().cuda()

    if global_loss:
        in_mat, out_mat = get_global_distance_map(train_embeddeds, train_targets, k, n_K)
        data_id = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=70, desc="Train")):
        data = data.cuda()
        target = target.cuda()

        # !note, for Triplet, here just the embeddings, for Clas, it's predict
        outputs = model(data)
        embeddeds = outputs

        if method == 'kTriplet':
            anchor, positive, negative = generate_k_triplet(embeddeds, target, K=k, B=n_K)
        elif method == 'batchHardTriplet':
            anchor, positive, negative = generate_batch_hard_triplet(embeddeds, target)
        elif method == 'batchAllTriplet' or method == 'batchSemiHardTriplet':
            anchor, positive, negative = generate_all_triplet(embeddeds, target)
        else:
            raise ValueError

        triplet_loss = loss_fn(anchor, positive, negative)

        if method == 'batchSemiHardTriplet':
            semi = torch.nonzero((triplet_loss <= loss_fn.margin) & (triplet_loss > 0))
            triplet_loss.index_select(dim=0, index=semi.squeeze())

        if rm_zero:
            non_zero = torch.nonzero(triplet_loss.cpu().data).size(0)
            if non_zero == 0:
                loss_triplet = triplet_loss.mean()
            else:
                loss_triplet = (triplet_loss / non_zero).sum()
        else:
            loss_triplet = triplet_loss.mean()

        if gamma > 0:
            loss_inner_class = torch.log1p((anchor - positive).pow(2).sum(1)).mean()
            loss = loss_triplet + gamma * loss_inner_class
        elif gamma == 0:
            loss_inner_class = np.mean(
                np.sum(np.power(np.log1p((anchor.cpu().detach().numpy() - positive.cpu().detach().numpy())), 2),
                       axis=1))
            loss_inner_class = torch.tensor(loss_inner_class)
            loss = loss_triplet
        else:
            loss_inner_class = torch.tensor(0)
            loss = loss_triplet

        if center_sigma > 0:
            closs = center_sigma * center_loss(embeddeds, target)
            loss += closs
            losses_center.append(closs.item())

        if global_loss:
            g_pos, g_neg = get_global_data(train_loader.dataset, in_mat, out_mat,
                                           np.array(range(data_id, data_id + train_loader.batch_size)))
            data_id += train_loader.batch_size
            g_pos = g_pos.cuda()
            g_neg = g_neg.cuda()
            out_p = model(g_pos)
            out_n = model(g_neg)

            anchor, positive, negative = makeup_global_triplet(embeddeds, out_p, out_n, k, n_K)
            global_loss = loss_fn(anchor, positive, negative)

            loss += global_loss

        # TODO
        # add cross entropy
        if use_cross_entropy:
            #ce_loss = cross_entropy(, target)
            #loss = loss + ce_loss
            print('NOt Implement!')
            sys.exit(-1)
            pass
        else:
            ce_loss = torch.tensor(0)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        losses_triplet.append(loss_triplet.item())
        losses_inner_class.append(loss_inner_class.item())
        losses_cross_entropy.append(ce_loss.item())

        losses.append(loss.item())

    average_triplet_loss = sum(losses_triplet) / len(train_loader)
    average_inner_class = sum(losses_inner_class) / len(train_loader)
    average_center_loss = sum(losses_center) / len(train_loader)
    average_cross_entropy = sum(losses_cross_entropy) / len(train_loader)
    total_loss = sum(losses) / len(train_loader)

    return total_loss, average_triplet_loss, average_inner_class, average_center_loss, average_cross_entropy


def test_epoch(test_loader, model, loss_fn, k, n_K, rm_zero, n_classes):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, ncols=70, desc="Test")):
            data = data.cuda()
            target = target.cuda()

            outputs = model(data)

            embeddeds = outputs

            anchor, positive, negative = generate_k_triplet(embeddeds, target, k, n_K)
            loss = loss_fn(anchor, positive, negative)
            if rm_zero:
                non_zero = torch.nonzero(loss.cpu().data).size(0)
                if non_zero == 0:
                    loss = loss.mean()
                else:
                    loss = (loss / non_zero).sum()
            else:
                loss = loss.mean()
            test_loss += loss.item()

        test_loss /= len(test_loader)

        return test_loss


def get_train_embeddeds(train_loader, model):
    with torch.no_grad():
        model.eval()

        train_embeddeds = []
        train_targets = []

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=70, desc="Get Train Samples")):
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            train_embeddeds.append(output.cpu().data)
            train_targets.append(target.cpu().data)

        train_embeddeds = torch.cat(train_embeddeds, 0)
        train_targets = torch.cat(train_targets, 0)

    return train_embeddeds, train_targets


def predictKNN(train_embeddeds, train_targets, test_loader, model, n_classes, k: list, train=False,
               data_augmentation=False):
    with torch.no_grad():
        model.eval()

        correct = []
        predicteds = {}
        for kk in k:
            predicteds[kk] = []

        index = 1 if train else 0

        for batch_idx, (data, target) in enumerate(tqdm(test_loader, ncols=70, desc="Predict")):
            data = data.cuda()
            batch_len = data.size(0)

            # if using five crop when test data augmentation, predict and get average vector
            if data_augmentation is True:
                bns, ncrops, c, h, w = data.size()
                output = model(data.view(-1, c, h, w))
                output_avg = output.view(bns, ncrops, -1).mean(1)
                output = output_avg
            else:
                output = model(data)

            train_embeddeds = train_embeddeds.cuda()
            dis_mat = pairwise_distances(output, train_embeddeds)

            dis_sort_id = torch.argsort(dis_mat, dim=1)

            for kk in k:
                closek_id = dis_sort_id[:, index: index + kk]
                closek_target = train_targets[closek_id].cpu().data.numpy()

                for i in range(batch_len):
                    cls_count = np.bincount(closek_target[i])
                    predicteds[kk].append(np.argmax(cls_count))

            target = target.numpy()
            correct.extend(target)

            # print("Predict KNN(K={}): [{}/{}]\t acc: {}/{}({:.2%})".format(k, batch_idx, len(test_loader), correct,
            #                                                                batch_len, correct / batch_len))

        accs, mcas, mcps, crs, cps = {}, {}, {}, {}, {}
        for kk in k:
            accuracy, mca, mcp, class_recall, class_precision = get_metrics(correct, predicteds[kk], n_classes)
            accs[kk], mcas[kk], mcps[kk] = accuracy, mca, mcp
            crs[kk], cps[kk] = class_recall, class_precision
        print("Predict: accuracy: {}, mca: {}, mcp: {}".format(accs, mcas, mcps))

        return accs, mcas, mcps, crs, cps


def get_global_distance_map(train_embeddeds, train_targets, k, n_k):
    """
    :param train_embeddeds:
    :param train_targets:
    :param k:
    :param n_k:
    :return: in_mat, out_mat
    """
    embeddeds_len = len(train_embeddeds)
    train_embeddeds = train_embeddeds.cuda()
    dis_mat = np.zeros((embeddeds_len, embeddeds_len))
    l = embeddeds_len // 100
    if embeddeds_len % 100 != 0:
        l += 1
    for i in tqdm(range(l), ncols=70, desc="Global Dismat"):
        s = i * 100
        dis_mat[:, s: s + 100] = pairwise_distances(train_embeddeds, train_embeddeds[s: s + 100]).cpu().data.numpy()
    ts = train_targets.data.numpy()

    in_mat = np.zeros((embeddeds_len, k))
    out_mat = np.zeros((embeddeds_len, n_k))

    for i in tqdm(range(embeddeds_len), ncols=70, desc="Global Dismat"):
        incls_id = np.where(ts == ts[i])[0]
        incls = dis_mat[i][incls_id]

        outcls_id = np.where(ts != ts[i])[0]
        outcls = dis_mat[i][outcls_id]

        incls_closeK = np.argsort(incls)[1:1 + k]
        outcls_closeB = np.argsort(outcls)[0: n_k]

        in_mat[i] = incls_id[incls_closeK]
        out_mat[i] = outcls_id[outcls_closeB]

    return in_mat, out_mat


def get_global_data(dataset, in_mat, out_mat, indexs):
    pos_data = []
    for i in indexs:
        if i >= len(in_mat):
            break
        in_ids = in_mat[i]
        for j in in_ids:
            pos_data.append(dataset[int(j)][0].unsqueeze(0))
    pos_data = torch.cat(pos_data, 0)

    neg_data = []
    for i in indexs:
        if i >= len(out_mat):
            break
        out_ids = out_mat[i]
        for j in out_ids:
            neg_data.append(dataset[int(j)][0].unsqueeze(0))
    neg_data = torch.cat(neg_data, 0)

    return pos_data, neg_data


def makeup_global_triplet(embeddeds, out_p, out_n, k, n_k):
    anchor = []
    negative = []
    positive = []
    for i, a in enumerate(embeddeds):
        a_p = out_p[i * k: (i + 1) * k]
        a_n = out_n[i * n_k: (i + 1) * n_k]
        an = a.unsqueeze(0)
        for p in a_p:
            for n in a_n:
                anchor.append(an)
                positive.append(p.unsqueeze(0))
                negative.append(n.unsqueeze(0))
    anchor = torch.cat(anchor, 0)
    positive = torch.cat(positive, 0)
    negative = torch.cat(negative, 0)
    return anchor, positive, negative
