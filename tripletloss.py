import argparse
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import data_loader
import datetime
import model_
from torch.backends import cudnn
import numpy as np
from PIL import ImageFile
from utils import TripletMarginLoss, PentaQLoss
from torch.autograd import Variable
from model import ClassificationNet,EmbeddingNet
from tensorboardX import SummaryWriter
from sklearn import neighbors
from utils import PairwiseDistance
import pandas as pd
import metrics
import shutil
# from increment_train import feature
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    parser = argparse.ArgumentParser(
        description='Classifiar using triplet loss.')
    parser.add_argument('--CVDs', type=str, default='0', metavar='CUDA_VISIBLE_DEVICES',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--server', type=int, default= 82, metavar='T',
                        help='which server is being used')
    parser.add_argument('--train-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/train2', metavar='dir',
                        help='path of train set.')
    parser.add_argument('--test-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/test2', metavar='dir',
                        help='path of test set.')
    parser.add_argument('--train-set-csv', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/train.csv', metavar='file',
                        help='path of train set.csv.')
    parser.add_argument('--num-triplet', type=int, default=1000, metavar='number',
                        help='number of triplet in dataset (default: 32)')
    parser.add_argument('--train-batch-size', type=int, default=96, metavar='number',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=192, metavar='number',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='number',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--embedding-size', type=int, default=128, metavar='number',
                        help='embedding size of model (default: 256)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--margin', type=float, default=1., metavar='margin',
                        help='loss margin (default: 1.0)')
    parser.add_argument('--num-classes', type=int, default=10, metavar='number',
                        help='classes number of dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='number',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, default='resnet34', metavar='M',
                        help='model name (default: resnet34)')
    parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                        help='Dropout probability (default: 0.2)')
    parser.add_argument('--check-path', type=str,default='/home/zili/memory/FaceRecognition-master/checkpoints', metavar='folder',
                        help='Checkpoint path')
    parser.add_argument('--is-semihard', type=bool, default=True, metavar='R',
                        help='whether the dataset is selected in semi-hard way.')
    parser.add_argument('--is-pretrained', type=bool, default=False,metavar='R',
                        help='whether model is pretrained.')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs
    if args.server == 31:
        args.train_set  = '/share/zili/code/triplet/data/cifar100/train2'
        args.test_set   = '/share/zili/code/triplet/data/cifar100/test2'
        args.train_set_csv = '/share/zili/code/triplet/data/cifar100/train.csv'
        args.check_path = '/share/zili/code/triplet/checkpoints'
    if args.server == 16:
        args.train_set = '/data0/zili/code/triplet/data/cifar100/train2'
        args.test_set = '/data0/zili/code/triplet/data/cifar100/test2'
        args.train_set_csv = '/data0/zili/code/triplet/data/cifar100/train.csv'
        args.check_path = '/data0/zili/code/triplet/checkpoints'
    if args.server == 17:
        args.train_set = '/data/jiaxin/zili/data/cifar100/train2'
        args.test_set = '/data/jiaxin/zili/data/cifar100/test'
        args.train_set_csv = '/data/jiaxin/zili/data/cifar100/train.csv'
        args.check_path = '/data/jiaxin/zili/checkpoints'
    now_time = str(datetime.datetime.now())
    if not os.path.exists(args.check_path):
        os.mkdir(args.check_path)
    args.check_path = os.path.join(args.check_path, now_time)
    if not os.path.exists(args.check_path):
        os.mkdir(args.check_path)
    shutil.copy('tripletloss.py', args.check_path)

    output1 = 'main_' + now_time
    f = open(args.check_path + os.path.sep + output1 + '.txt', 'w+')
    writer = SummaryWriter()

    print('Loading model...')

    # model = FaceModel(model_name     = args.model_name,
    #                   embedding_size = args.embedding_size,
    #                   pretrained     = args.is_pretrained)
    # model = model_.ResNet34(True, args.embedding_size)
    model = EmbeddingNet(embedding_len=args.embedding_size)
    if torch.cuda.is_available():

        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    f.write("     model: {}".format(model.module) + '\r\n')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    print('start training...')

    features, labels, clf, destination = select_three_sample(model, args, -1, writer)
    for epoch in range(args.epochs):
        file_operation(f, args, optimizer)

        if (epoch + 1) % 10 == 0 :
            args.lr = args.lr / 3
            update_lr(optimizer, args.lr)

        train(epoch, model, optimizer, args, f, features, destination, writer)
        features, labels, clf, destination = select_three_sample(model, args, epoch, writer)
        validate(epoch, model, clf, args, f, writer, False)
        validate(epoch, model, clf, args, f, writer, True)

        f.write('\r\n')
        if epoch < 80 and epoch > 10:
            torch.save(model, args.check_path + os.path.sep + 'epoch' + str(epoch) + '.pth')
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def train(epoch, model, optimizer, args, f, features, destination, writer):
    model.train()

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.4914, 0.4822, 0.4465]),
            std=np.array([0.2023, 0.1994, 0.2010])),
    ])

    if args.is_semihard:
        train_set = data_loader.SemiHardTripletDataset(root_dir     = args.train_set,
                                                       csv_name     = args.train_set_csv,
                                                       num_triplets = args.num_triplet,
                                                       features     = features,
                                                       margin       = args.margin,
                                                       destination  = destination,
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
    for batch_idx, (data, target) in enumerate(train_loader):

        data[0], target[0] = data[0].cuda(), target[0].cuda()
        data[1], target[1] = data[1].cuda(), target[1].cuda()
        data[2], target[2] = data[2].cuda(), target[2].cuda()
        # data[3], target[3] = data[3].cuda(), target[3].cuda()
        # data[4], target[4] = data[4].cuda(), target[4].cuda()

        data[0], target[0] = Variable(data[0]), Variable(target[0])
        data[1], target[1] = Variable(data[1]), Variable(target[1])
        data[2], target[2] = Variable(data[2]), Variable(target[2])
        # data[3], target[3] = Variable(data[3]), Variable(target[3])
        # data[4], target[4] = Variable(data[4]), Variable(target[4])

        optimizer.zero_grad()

        anchor   = model.forward(data[0])
        positive = model.forward(data[1])
        negative = model.forward(data[2])
        # anc_mea  = model.forward(data[3])
        # neg_mea  = model.forward(data[4])
        loss = TripletMarginLoss(margin=args.margin, num_classes=args.num_classes).forward(anchor, positive, negative)
        # loss = PentaQLoss(margin=args.margin).forward(anchor, positive, negative, anc_mea, neg_mea)
        total_loss += loss.item()

        loss.backward()
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


def validate(epoch, model, clf, args, f, writer, test = True):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])
    test_set = torchvision.datasets.ImageFolder(root = args.test_set if test else args.train_set, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.test_batch_size, shuffle = True)

    correct, total = 0, 0
    target_arr, predict_arr = [], []
    fea, l = torch.zeros(0), torch.zeros(0)

    data1 = [0]*args.num_classes
    data2 = [0]*args.num_classes
    data3 = [0]*args.num_classes

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model.forward(data)
        predicted = clf.predict(output.data.cpu().numpy())
        predict_arr.append(predicted)
        target_arr.append(target.data.cpu().numpy())
        fea = torch.cat((fea, output.data.cpu()))
        l = torch.cat((l, target.data.cpu().float()))
        for i in range(0, target.size(0)):
            data1[target[i]] += 1
            data3[predicted[i]] += 1
            if target[i] == predicted[i]:
                data2[target[i]] += 1

        correct += (torch.tensor(predicted) == target.data.cpu()).sum()
        total += target.size(0)

        writer.add_scalar('/Acc', 100. * correct / float(total), epoch * len(test_loader) + i)

    k_n = (epoch * 2) if test else (epoch * 2 + 1)
    writer.add_embedding(mat=fea, metadata=l, global_step=k_n)

    cm_path = args.check_path + '/' + 'train' + str(epoch) + '_confusematrix'
    if test:
        cm_path = args.check_path + '/' + 'test' + str(epoch) + '_confusematrix'
    cm = metrics.get_confuse_matrix(predict_arr, target_arr)
    np.save(cm_path, cm)

    for j in range(args.num_classes):
        recall = 0 if data1[j] == 0 else data2[j] / data1[j]
        precision = 0 if data3[j] == 0 else data2[j] / data3[j]
        context = 'Class%1s: recall is %.2f%% (%d in %d), precision is %.2f%% (%d in %d)' % (
            str(j), 100 * recall, data2[j], data1[j],
            100 * precision, data2[j], data3[j])
        print(context)
        f.write(context + "\r\n")

    context = 'Accuracy of model in ' + ('test' if test else 'train') + \
              ' set is {}/{}({:.2f}%)'.format(correct, total, 100. * float( correct) / float(total))
    f.write(context + '\r\n')
    print(context)


def select_three_sample(model, args, epoch, writer):
    model.eval()
    num_each_class = [500,500,500,500,500,500,500,50,50,50]
    # num_each_class = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])
    train_set = torchvision.datasets.ImageFolder(root=args.train_set, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.test_batch_size, shuffle=False)

    NearestCentroid, KNeighbors, features, label, labels =[], [], [], [], []
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        features.extend(output.data)
        KNeighbors.extend(output.data.cpu().numpy())
        labels.extend(target.data.cpu().numpy())

    count = 0
    l2_dist = PairwiseDistance(2)
    destination = os.path.join(args.check_path, 'epoch'+str(epoch))
    if not os.path.exists(destination):
        os.mkdir(destination)
    for i in range(len(num_each_class)):
        num_sample = features[count:count + num_each_class[i]]
        m = torch.tensor(np.zeros(args.embedding_size)).float().cuda()
        for x in num_sample:
            m += x
        m /= num_each_class[i]

        sample1 = min(num_sample, key=lambda x:l2_dist.forward_val(x, m))
        sample2 = max(num_sample, key=lambda x:l2_dist.forward_val(x, sample1))
        sample3 = max(num_sample, key=lambda x:l2_dist.forward_val(x, sample2))
        NearestCentroid.append(sample1.cpu().numpy())
        label.append(i)

        sample1_loc, sample2_loc, sample3_loc = -1, -1, -1
        for j in range(num_sample.__len__()):
            if (num_sample[j] == sample1).all():
                sample1_loc = j
            if (num_sample[j] == sample2).all():
                sample2_loc = j
            if (num_sample[j] == sample3).all():
                sample3_loc = j

        frame = pd.read_csv(args.train_set_csv)
        destination_class = os.path.join(destination, str(frame['name'][count+sample1_loc]))
        if not os.path.exists(destination_class):
            os.mkdir(destination_class)
        sample1_source = os.path.join(args.train_set, str(frame['name'][count+sample1_loc]), str(frame['id'][count+sample1_loc]) + '.png')
        sample2_source = os.path.join(args.train_set, str(frame['name'][count+sample2_loc]), str(frame['id'][count+sample2_loc]) + '.png')
        sample3_source = os.path.join(args.train_set, str(frame['name'][count+sample3_loc]), str(frame['id'][count+sample3_loc]) + '.png')
        shutil.copy(sample1_source, destination_class+'/sample1.png')
        shutil.copy(sample2_source, destination_class+'/sample2.png')
        shutil.copy(sample3_source, destination_class+'/sample3.png')
        count += num_each_class[i]

    clf = neighbors.NearestCentroid()
    clf.fit(NearestCentroid, label)

    return features, labels, clf, destination

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def file_operation(f,args,optimizer):
    print("train dataset:{}".format(args.train_set))
    print("dropout: {}".format(args.dropout_p))
    print("margin:{}".format(args.margin))
    print("is semi-hard: {}".format(args.is_semihard))
    print("model: {}".format(args.model_name))
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
    f.write("model: {}".format(args.model_name) + '\r\n')
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

