import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import data_loader
import datetime
from torch.backends import cudnn
import numpy as np
from PIL import ImageFile
from utils import PairwiseDistance, TripletMarginLoss, generate_csv
from torch.autograd import Variable
from model import FaceModel
from tensorboardX import SummaryWriter
from sklearn import neighbors
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    parser = argparse.ArgumentParser(
        description='Classifiar using triplet loss.')
    parser.add_argument('--CVDs', type=str, default='0,1,2,3', metavar='CUDA_VISIBLE_DEVICES',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--train-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/mnist/train', metavar='dir',
                        help='path of train set.')
    parser.add_argument('--test-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/mnist/test', metavar='dir',
                        help='path of test set.')
    parser.add_argument('--train-set-csv', type=str, default='/home/zili/memory/FaceRecognition-master/data/mnist/train.csv', metavar='file',
                        help='path of train set.csv.')
    parser.add_argument('--test-set-csv', type=str, default='/home/zili/memory/FaceRecognition-master/data/mnist/test.csv', metavar='file',
                        help='path of test set.csv.')
    parser.add_argument('--num-triplet', type=int, default=10000, metavar='N',
                        help='number of triplet in dataset (default: 32)')
    parser.add_argument('--train-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--embedding-size', type=int, default=256, metavar='N',
                        help='embedding size of model (default: 256)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--margin', type=float, default=1.0, metavar='margin',
                        help='loss margin (default: 1.0)')
    parser.add_argument('--kneighbor', type=int, default=20, metavar='N',
                        help='how many neighbor in testing')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='classes number of dataset')
    parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, default='resnet34', metavar='M',
                        help='model name (default: resnet34)')
    parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                        help='Dropout probability (default: 0.2)')
    parser.add_argument('--check-path', type=str,default='checkpoints3', metavar='C',
                        help='Checkpoint path')
    parser.add_argument('--is-semihard', type=bool, default=True, metavar='R',
                        help='whether the dataset is selected in semi-hard way.')
    parser.add_argument('--is-pretrained', type=bool, default=False,metavar='R',
                        help='whether model is pretrained.')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs

    output1 = 'main' + str(datetime.datetime.now())
    f = open(args.check_path + os.path.sep + output1 + '.txt', 'w+')

    l2_dist = PairwiseDistance(2)
    writer = SummaryWriter()

    print('Loading model...')

    model = FaceModel(embedding_size = args.embedding_size,
                      num_classes    = args.num_classes,
                      pretrained     = args.is_pretrained)
    f.write("            model: {}".format(model.model) + '\r\n')

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('start training...')
    features, labels, clf = feature(model, args)

    for epoch in range(args.epochs):
        if epoch % 5 == 0:
            file_operation(f, args, optimizer)

        if (epoch + 1) % 2 == 0 :
            args.lr = args.lr / 3
            update_lr(optimizer, args.lr)

        generate_csv(args)
        train(epoch, model, optimizer, args, f, writer, features)
        features, labels, clf = feature(model, args)
        validate(epoch, model, clf, args, f, writer)

        f.write('\r\n')
    torch.save(model, args.check_path + os.path.sep + output1 + '.pkl')


def train(epoch, model, optimizer, args, f, writer, features):
    model.train()

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])

    if args.is_semihard:
        train_set = data_loader.SemiHardTripletDataset(root_dir     = args.train_set,
                                                       csv_name     = args.train_set_csv,
                                                       num_triplets = args.num_triplet,
                                                       features     = features,
                                                       margin       = args.margin,
                                                       transform    = transform)
    else:
        train_set = data_loader.TripletFaceDataset(root_dir     = args.train_set,
                                                   csv_name     = args.train_set_csv,
                                                   num_triplets = args.num_triplet,
                                                   transform    = transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size  = args.train_batch_size,
                                               shuffle     = True,
                                               num_workers = 2)

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):

        data[0], target[0] = data[0].cuda(), target[0].cuda()
        data[1], target[1] = data[1].cuda(), target[1].cuda()
        data[2], target[2] = data[2].cuda(), target[2].cuda()

        data[0], target[0] = Variable(data[0]), Variable(target[0])
        data[1], target[1] = Variable(data[1]), Variable(target[1])
        data[2], target[2] = Variable(data[2]), Variable(target[2])

        optimizer.zero_grad()

        anchor   = model.forward(data[0])
        positive = model.forward(data[1])
        negative = model.forward(data[2])

        loss = TripletMarginLoss(margin = args.margin).forward(anchor, positive, negative)
        loss_entropy1 = criterion(model.module.forward_classifier(anchor),   target[0].squeeze().long())
        loss_entropy2 = criterion(model.module.forward_classifier(positive), target[1].squeeze().long())
        loss_entropy3 = criterion(model.module.forward_classifier(negative), target[2].squeeze().long())
        loss_entropy  = loss_entropy1 + loss_entropy2 + loss_entropy3
        total_loss += loss.item()

        loss.backward(retain_graph=True)
        loss_entropy.backward()
        optimizer.step()

        writer.add_scalar('/Loss', loss.item(), epoch * len(train_loader) + batch_idx)

        if (batch_idx+1) % args.log_interval == 0:
            context = 'Train Epoch: {} [{}/{} ({:.0f}%)], Average loss: {:.4f}'.format(
                epoch, batch_idx * len(data[0]), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), total_loss / (batch_idx + 1))
            print(context)
            f.write(context + '\r\n')
    context = 'Train Epoch: {} [{}/{} ({:.0f}%)], Average loss: {:.4f}'.format(
        epoch, len(train_loader.dataset), len(train_loader.dataset),
        100.0 * len(train_loader) / len(train_loader), total_loss / len(train_loader))
    print(context)
    f.write(context + '\r\n')
    torch.save(model, args.check_path + os.path.sep  + 'epoch{}.pkl'.format(epoch))


def validate(epoch, model, clf, args, f, writer):
    model.eval()

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])
    test_set = torchvision.datasets.ImageFolder(root = args.test_set, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.test_batch_size, shuffle = True)
    correct = 0
    total = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model.forward(data)
        # predicted = clf.predict(output.data.cpu().numpy())
        # correct += (torch.tensor(predicted) == target.data.cpu()).sum()
        output = model.module.forward_classifier(output)
        _, predicted = output.data.max(1)
        correct += predicted.eq(target).sum().item()

        total += target.size(0)

        writer.add_scalar('/Acc', 100. * correct / float(total), epoch * len(test_loader) + i)
        if (i + 1) % 10 == 0:
            context = 'Accuracy of model is {}/{}({:.3f}%)'.format(correct, total, 100. * float(correct) / float(total))
            f.write(context + '\r\n')
            print(context)


def feature(model, args):
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])
    train_set = torchvision.datasets.ImageFolder(root = args.train_set, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.test_batch_size, shuffle = False)

    features, labels = [], []

    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model.forward(data)
        features.extend(output.data.cpu().numpy())
        labels.extend(target.data.cpu().numpy())

    clf = neighbors.KNeighborsClassifier(n_neighbors = args.kneighbor)
    clf.fit(features, labels)
    return features, labels, clf


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def file_operation(f,args,optimizer):
    print("train dataset:{}".format(args.train_set))
    print("dropout: {}".format(args.dropout_p))
    print("margin:{}".format(args.margin))
    print("is semi-hard: {}".format(args.is_semihard))
    print("num_triplet: {}".format(args.num_triplet))
    print("check_path: {}".format(args.check_path))
    print("learing_rate: {}".format(args.lr))
    print("train_batch_size: {}".format(args.train_batch_size))
    print("test_batch_size: {}".format(args.test_batch_size))
    print("is_pretrained: {}".format(args.is_pretrained))
    print("optimizer: {}".format(optimizer))
    f.write("train dataset:{}".format(args.train_set) + '\r\n')
    f.write("dropout: {}".format(args.dropout_p) + '\r\n')
    f.write("margin:{}".format(args.margin) + '\r\n')
    f.write("is semi-hard: {}".format(args.is_semihard) + '\r\n')
    f.write("num_triplet: {}".format(args.num_triplet) + '\r\n')
    f.write("check_path: {}".format(args.check_path) + '\r\n')
    f.write("learing_rate: {}".format(args.lr) + '\r\n')
    f.write("train_batch_size: {}".format(args.train_batch_size) + '\r\n')
    f.write("test_batch_size: {}".format(args.test_batch_size) + '\r\n')
    f.write("is_pretrained: {}".format(args.is_pretrained) + '\r\n')
    f.write("optimizer: {}".format(optimizer) + '\r\n')


if __name__ == '__main__':
    main()

