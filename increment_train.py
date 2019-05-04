import argparse
import os
import metrics
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import data_loader
import dataset
import datetime
from torch.backends import cudnn
import numpy as np
from PIL import ImageFile
from utils import PairwiseDistance, TripletMarginLoss, generate_csv
from torch.autograd import Variable
from model import ClassificationNet, EmbeddingNet
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorboardX import SummaryWriter
from sklearn import neighbors
import shutil
def main():
    parser = argparse.ArgumentParser(
        description='Classifiar using triplet loss.')
    parser.add_argument('--CVDs', type=str, default='4,5', metavar='CUDA_VISIBLE_DEVICES',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--server', type=int, default=82, metavar='T',
                        help='which server is being used')
    parser.add_argument('--train-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/increment', metavar='dir',
                        help='path of train set.')
    parser.add_argument('--test-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/test2',
                        metavar='dir', help='path of test set.')
    parser.add_argument('--preserved-sample', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/test2',
                        metavar='dir', help='path of test set.')
    parser.add_argument('--train-set-csv', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/increase_train.csv', metavar='file',
                        help='path of train set.csv.')
    parser.add_argument('--num-triplet', type=int, default=10000, metavar='N',
                        help='number of triplet in dataset (default: 32)')
    parser.add_argument('--train-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--embedding-size', type=int, default=256, metavar='N',
                        help='embedding size of model (default: 256)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--margin', type=float, default=1.0, metavar='margin',
                        help='loss margin (default: 1.0)')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='classes number of dataset')
    parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=16, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, default='resnet34', metavar='M',
                        help='model name (default: resnet34)')
    parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                        help='Dropout probability (default: 0.2)')
    parser.add_argument('--check-path', type=str,default='/home/zili/memory/FaceRecognition-master/checkpoints3', metavar='C',
                        help='Checkpoint path')
    parser.add_argument('--semi-hard', type=bool, default=False, metavar='R',
                        help='whether the dataset is selected in semi-hard way.')
    parser.add_argument('--pretrained', type=bool, default=False, metavar='R',
                        help='whether model is pretrained.')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs

    if args.server == 31:
        args.train_set  = '/share/zili/code/triplet/data/cifar100/increment'
        args.test_set   = '/share/zili/code/triplet/data/cifar100'
        args.train_set_csv = '/share/zili/code/triplet/data/cifar100/increase_train.csv'
        args.check_path = '/share/zili/code/triplet/checkpoints3'
        args.preserved_sample = '/share/zili/code/triplet/checkpoints/2019-03-22 19:17:53.624217/epoch17'
    if args.server == 16:
        args.train_set  = '/data0/zili/code/triplet/data/cifar100/increment'
        args.test_set   = '/data0/zili/code/triplet/data/cifar100'
        args.train_set_csv = '/data0/zili/code/triplet/data/cifar100/increase_train.csv'
        args.check_path = '/data0/zili/code/triplet/checkpoints3'
        args.preserved_sample = '/data0/zili/code/triplet/checkpoints/2019-03-18 23:39:43.943989/epoch40'
    if args.server == 17:
        args.train_set = '/data/jiaxin/zili/data/cifar100/increment'
        args.test_set = '/data/jiaxin/zili/data/cifar100'
        args.train_set_csv = '/data/jiaxin/zili/data/cifar100/increase_train.csv'
        args.check_path = '/data/jiaxin/zili/checkpoints3'
        args.preserved_sample = '/data/jiaxin/zili/checkpoints/2019-03-28 17:15:19.936335/epoch58'
    if args.server == 15:
        args.train_set = '/home/zili/code/triplet/data/cifar100/train2'
        args.test_set = '/home/zili/code/triplet/data/cifar100/test2'
        args.train_set_csv = '/home/zili/code/triplet/data/cifar100/train.csv'
        args.check_path = '/home/zili/code/triplet/checkpoints'
    now_time = str(datetime.datetime.now())
    args.check_path = os.path.join(args.check_path, now_time)

    os.mkdir(args.check_path)

    f = open(args.check_path + os.path.sep + now_time + '.txt', 'w+')
    shutil.copy('increment_train.py', args.check_path)

    writer = SummaryWriter()

    print('Loading model...')
    model = torch.load(args.preserved_sample+'.pth')
    # print(model)
    f.write("            model: {}".format(model) + '\r\n')

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('start training...')
    features, labels, clf, preserved_features, preserved_labels = feature(model, args, writer, -1)
    validate(-1, model, clf, args, f, writer)

    for epoch in range(args.epochs):
        file_operation(f, args, optimizer)

        if (epoch + 1) % 2 == 0 :
            args.lr = args.lr / 3
            update_lr(optimizer, args.lr)

        train(epoch, model, optimizer, args, f, features, preserved_features, writer)
        features, labels, clf, preserved_features, preserved_labels = feature(model, args, writer, epoch)
        validate(epoch, model, clf, args, f, writer, test=False)
        validate(epoch, model, clf, args, f, writer)

        f.write('\r\n')
        if epoch < 100 and epoch > 10:
            torch.save(model, args.check_path + os.path.sep + 'epoch' + str(epoch) + '.pth')
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def train(epoch, model, optimizer, args, f, features, preserved_features, writer):
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

    if args.semi_hard:
        train_set = dataset.SemiHardTripletDataset(root_dir        = args.train_set,
                                                    csv_name       = args.train_set_csv,
                                                root_dir_preserved = args.preserved_sample,
                                                   preserved_sample= args.preserved_sample+'/preserved_train.csv',
                                                    num_triplets   = args.num_triplet,
                                                       features    = features,
                                                preserved_features = preserved_features,
                                                    margin         = args.margin,
                                                    transform      = transform)
        # train_set = data_loader.SemiHardTripletDataset(root_dir=args.train_set,
        #                                                csv_name=args.train_set_csv,
        #                                                num_triplets=args.num_triplet,
        #                                                features=features,
        #                                                margin=args.margin,
        #                                                transform=transform)
    else:
        train_set = dataset.TripletFaceDataset(root_dir     = args.train_set,
                                                csv_name     = args.train_set_csv,
                                               root_dir_preserved= args.preserved_sample,
                                               preserved_sample= args.preserved_sample+'/preserved_train.csv',
                                                   num_triplets = args.num_triplet,
                                                   transform    = transform)

        # train_set = data_loader.TripletFaceDataset(root_dir=args.train_set,
        #                                            csv_name=args.train_set_csv,
        #                                            num_triplets=args.num_triplet,
        #                                            transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size  = args.train_batch_size,
                                               shuffle     = True,
                                               num_workers = 2)

    total_loss = 0.0
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
    torch.save(model, args.check_path + os.path.sep  + 'epoch{}.pth'.format(epoch))


def validate(epoch, model, clf, args, f, writer, test=True):
    model.eval()

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])
    dir = args.test_set + '/test'
    test_set = torchvision.datasets.ImageFolder(root = dir if test else args.train_set, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.test_batch_size, shuffle = True)
    correct = 0
    total = 0
    target_arr = []
    predict_arr = []
    data1 = [0]*args.num_classes
    data2 = [0]*args.num_classes
    data3 = [0]*args.num_classes
    fea, l = torch.zeros(0), torch.zeros(0)
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

    if not test:
        test_set = torchvision.datasets.ImageFolder(root=args.test_set + '/train2', transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)
        for i, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model.forward(data)
            fea = torch.cat((fea, output.data.cpu()))
            l = torch.cat((l, target.data.cpu().float()))

    cm_path = args.check_path + '/' + 'test' + str(epoch) + '_confusematrix'
    k_n = (epoch * 3 + 2) if test else (epoch * 3 + 1)
    # if epoch > -1:
    #     writer.add_embedding(mat=fea, metadata=l, global_step=k_n)
    cm = metrics.get_confuse_matrix(predict_arr, target_arr)
    np.save(cm_path, cm)
    for j in range(10):
        recall = 0 if data1[j] == 0 else data2[j] / data1[j]
        precision = 0 if data3[j] == 0 else data2[j] / data3[j]
        context = 'Class%1s: recall is %.2f%% (%d in %d), precision is %.2f%% (%d in %d)' % (
            str(j), 100 * recall, data2[j], data1[j],
            100 * precision, data2[j], data3[j])
        print(context)
        f.write(context + "\r\n")
    context = 'Accuracy of model is {}/{}({:.2f}%)'.format(correct, total, 100. * float(correct) / float(total))
    f.write(context + '\r\n')
    print(context)

def feature(model, args, writer, epoch):
    model.eval()
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])

    NearestCentroid, NearestCentroid_label, features, labels = [], [], [], []
    preserved_features, preserved_labels =[], []
    fea, l = torch.zeros(0), torch.zeros(0)

    train_set = torchvision.datasets.ImageFolder(root=args.preserved_sample, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=3, shuffle=False)
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model.module.forward(data)
        preserved_features.extend(output.data)
        preserved_labels.extend(target.data.cpu().numpy())
        NearestCentroid.append(output[0].data.cpu().numpy())
        NearestCentroid_label.append(target[0].data.cpu().numpy())
        fea = torch.cat((fea, output.data.cpu()))
        l = torch.cat((l, target.data.cpu().float()))


    train_set = torchvision.datasets.ImageFolder(root= args.train_set, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.test_batch_size, shuffle=False)
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model.forward(data)
        features.extend(output.data)
        labels.extend(target.data.cpu().numpy())
        fea = torch.cat((fea, output.data.cpu()))
        l = torch.cat((l, target.data.cpu().float()))

    clf = neighbors.NearestCentroid()
    clf.fit(NearestCentroid, NearestCentroid_label)


    writer.add_embedding(mat=fea, metadata=l, global_step=epoch)
    return features, labels, clf, preserved_features, preserved_labels


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def file_operation(f,args,optimizer):
    print("train dataset:{}".format(args.train_set))
    print("dropout: {}".format(args.dropout_p))
    print("margin:{}".format(args.margin))
    print("semi-hard: {}".format(args.semi_hard))
    print("num_triplet: {}".format(args.num_triplet))
    print("check_path: {}".format(args.check_path))
    print("learing_rate: {}".format(args.lr))
    print("train_batch_size: {}".format(args.train_batch_size))
    print("test_batch_size: {}".format(args.test_batch_size))
    print("pretrained: {}".format(args.pretrained))
    print("optimizer: {}".format(optimizer))
    f.write("train dataset:{}".format(args.train_set) + '\r\n')
    f.write("dropout: {}".format(args.dropout_p) + '\r\n')
    f.write("margin:{}".format(args.margin) + '\r\n')
    f.write("semi-hard: {}".format(args.semi_hard) + '\r\n')
    f.write("num_triplet: {}".format(args.num_triplet) + '\r\n')
    f.write("check_path: {}".format(args.check_path) + '\r\n')
    f.write("learing_rate: {}".format(args.lr) + '\r\n')
    f.write("train_batch_size: {}".format(args.train_batch_size) + '\r\n')
    f.write("test_batch_size: {}".format(args.test_batch_size) + '\r\n')
    f.write("pretrained: {}".format(args.pretrained) + '\r\n')
    f.write("optimizer: {}".format(optimizer) + '\r\n')


if __name__ == '__main__':
    main()

