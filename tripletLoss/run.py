"""Main function to run model"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter


from networks import EmbeddingNet, ClassificationNet
from trainer import fit
from datasets import SpecificDataset, SampledDataset
from plot import extract_embeddings, plot_embeddings
from train_classification import fit_classification
from BatchSampler import BatchSampler
from losses import AlteredTripletLoss

torch.multiprocessing.set_sharing_strategy('file_system')

def str_to_bool(val):
    """convert str to bool"""
    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        raise ValueError


parser = argparse.ArgumentParser(description='Can you MICCAI')
parser.add_argument('--cuda', default='0',
                    help="cuda visible device")
parser.add_argument('--n_epochs', type=int, default=50,
                    help="all epochs")
parser.add_argument('--batch_size', type=int, default=96,
                    help='input batch size for training (default: 128),\
                    if using sampler, batch_size would be fake')
parser.add_argument('--lr', default=0.01, type=float,
                    help="lr")
parser.add_argument('--K', default=9, type=int,
                    help="K")
parser.add_argument('--n_K', default=9, type=int,
                    help='negative dot')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help="MNIST, cifar10, cifar100, SkinLesion, miniImageNet")
parser.add_argument('--pretrained', default=True, type=str_to_bool,
                    help="Fasle, True available")
parser.add_argument('--method', default='kTriplet', type=str,
                    help='kTriplet, classification, batchHardTriplet, batchAllTriplet, batchSemiHardTriplet available')
parser.add_argument('--log_dir', default='9999', type=str,
                    help='log dir name')
parser.add_argument('--amount', default=0, type=int,
                    help='amount of each class for train data')
parser.add_argument('--margin', default=1, type=float,
                    help='margin for triplet loss')
parser.add_argument('--optimizer', default='SGD', type=str,
                    help='optimizer')
parser.add_argument('--step_size', default='30', type=int,
                    help='Scheduler step size for SGD')
parser.add_argument('--global_loss', default='False', type=str_to_bool,
                    help='add global loss')
parser.add_argument('--shuffle_interval', default=5, type=int,
                    help='-1: do not shuffle, 0: always shuffle, \
                            other positive num: shuffle interval')
parser.add_argument('--triplet_loss_p', default=2, type=int,
                    help='triplet loss p, p=1,2,3...')
parser.add_argument('--network', default='resnet50', type=str,
                    help='resnet50, resnet110, cifar_resnet50')
parser.add_argument('--embedding_len', default=128, type=int,
                    help='128, 512, 1024, 2048')
parser.add_argument('--batch_n_classes', default=100, type=int,
                    help='depend on your dataset')
parser.add_argument('--batch_n_num', default=11, type=int,
                    help='depend on your dataset, number for each class per batch')
parser.add_argument('--use_sampler', default=True, type=str_to_bool,
                    help='whether to use sampler, note n_epoch, step_size should be bigger')
parser.add_argument('--gamma', default=0, type=float,
                    help='coeffient for inner class loss')
parser.add_argument('--rm_zero', default='True', type=str_to_bool,
                    help='True or False')
parser.add_argument('--weight_decay', default='False', type=str_to_bool,
                    help='True or False')
parser.add_argument('--data_augmentation', default=False, type=str_to_bool,
                    help='whether use data augentation')
parser.add_argument('--save_model_path', default='../model/', type=str,
                    help='path to save model')
parser.add_argument('--center_sigma', default=-1, type=float,
                    help='coeffient for center loss')
parser.add_argument('--freeze_parameter', default=False, type=str_to_bool,
                    help='True, False')
parser.add_argument('--use_cross_entropy', default=False, type=str_to_bool,
                    help='True, False')
# if use global loss, shuffle_interval must be not zero

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


def print_config():
    """print hyper parameters and basic setting"""
    print('=' * 20, 'basic setting start', '=' * 20)
    for arg in vars(args):
        print('{:15}: {}'.format(arg, getattr(args, arg)))
    print('=' * 20, 'basic setting end', '=' * 20)


def get_optimizer(opt, model, lr, weight_decay):
    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt == 'Adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=lr, final_lr=0.1)
    elif opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-3)
    else:
        print('No optim availabel')
        sys.exit(-1)
    return optimizer


def main():
    """main function"""

    writer = SummaryWriter(log_dir='runs/' + args.log_dir)  # tensorboard

    # hyper parameters setting
    lr = args.lr
    k = args.K
    amount = args.amount
    n_epochs = args.n_epochs
    log_interval = 100
    batch_size = args.batch_size
    pretrained = args.pretrained
    method = args.method
    n_K = args.n_K
    margin = args.margin
    shuffle_interval = args.shuffle_interval
    opt = args.optimizer
    step_size = args.step_size
    global_loss = args.global_loss
    triplet_loss_p = args.triplet_loss_p
    network = args.network
    embedding_len = args.embedding_len
    batch_n_classes = args.batch_n_classes
    batch_n_num = args.batch_n_num
    use_sampler = args.use_sampler
    rm_zero = args.rm_zero
    center_sigma = args.center_sigma
    gamma = args.gamma
    weight_decay = args.weight_decay
    data_augmentation = args.data_augmentation
    save_model_path = args.save_model_path
    log_dir = args.log_dir
    freeze_parameter = args.freeze_parameter
    use_cross_entropy = args.use_cross_entropy

    # load data
    dataset = SpecificDataset(args.dataset, data_augmentation)
    n_classes = dataset.n_classes
    classes = dataset.classes
    channels = dataset.channels
    width, height = dataset.width, dataset.height
    gap = dataset.gap

    train_dataset = SampledDataset(dataset.train_dataset, channels, amount)
    print('Train data has {}'.format(len(train_dataset)))

    test_dataset = dataset.test_dataset
    print('Validation data has {}'.format(len(test_dataset)))

    test_dataset_fc = dataset.test_dataset_fc if dataset.test_dataset_fc is not None else None
    kwargs = {'num_workers': 8, 'pin_memory': False}
    # tarin_shuffle = True if shuffle_interval == 0 else False
    tarin_shuffle = (shuffle_interval == 0)

    batch_sampler = BatchSampler(train_dataset, n_classes=batch_n_classes, n_num=batch_n_num)
    sampler_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_sampler=batch_sampler, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=tarin_shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False, **kwargs)
    test_fc_loader = torch.utils.data.DataLoader(test_dataset_fc,
                                                 batch_size=batch_size, shuffle=False,
                                                 **kwargs) if test_dataset_fc is not None else None

    embedding_net = EmbeddingNet(network=network, pretrained=pretrained,
                                 embedding_len=embedding_len, gap=gap, freeze_parameter=freeze_parameter)

    if method == 'classification':
        # model = resnet.resnet32().cuda()
        model = ClassificationNet(embedding_net, n_classes=n_classes, embedding_len=embedding_len).cuda()
    elif method in ['kTriplet', 'batchHardTriplet', 'batchAllTriplet', 'batchSemiHardTriplet']:
        model = embedding_net.cuda()
    else:
        print('method must provide')
        sys.exit(-1)


    optimizer = get_optimizer(opt, model, lr, weight_decay)

    if opt == 'SGD':
        #if args.dataset == 'SD198':
            #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500, 950], gamma=0.5, last_epoch=-1)
        #else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma=0.5, last_epoch=-1)
    else:
        scheduler = None

    # add model graph into tensorboard
    #dummy_input = torch.zeros(size=(batch_size, channels, height, width)).cuda()
    #writer.add_graph(model, dummy_input)
    #del dummy_input

    if method == 'classification':
        loss_fn = nn.CrossEntropyLoss().cuda()
        fit_classification(train_loader, test_loader, test_fc_loader, model, loss_fn, optimizer, scheduler, n_epochs,
                           writer=writer, n_classes=n_classes, data_augmentation=data_augmentation)

    elif method in ['kTriplet', 'batchHardTriplet', 'batchAllTriplet', 'batchSemiHardTriplet']:
        loss_fn = nn.TripletMarginLoss(margin=margin, p=triplet_loss_p, reduction='none').cuda()
        fit(train_loader, sampler_train_loader, test_loader, test_fc_loader, model, loss_fn, optimizer, scheduler,
            n_epochs, k, n_K, log_interval, shuffle_interval, global_loss=global_loss, writer=writer,
            n_classes=n_classes, gamma=gamma, center_sigma=center_sigma, use_sampler=use_sampler, rm_zero=rm_zero,
            method=method, data_augmentation=data_augmentation, freeze_parameter=freeze_parameter, use_cross_entropy=use_cross_entropy)

    # save model
    save_model_path = os.path.join(save_model_path, log_dir)
    torch.save(model.state_dict(), save_model_path)
    print('save model in {}'.format(save_model_path))

    # plot tensor in tensorboard
    train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model, embedding_len)
    plot_embeddings(train_embeddings_tl, train_labels_tl, classes, writer, tag='train_embeddings')
    val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model, embedding_len)
    plot_embeddings(val_embeddings_tl, val_labels_tl, classes, writer, tag='val_embeddings')


if __name__ == '__main__':
    print_config()
    main()
