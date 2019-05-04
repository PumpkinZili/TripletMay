import argparse
import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
# from cnn_finetune import make_model
import torch.nn as nn
import torch.optim as optim
import datetime
import model_
from torch.backends import cudnn
import numpy as np
from PIL import ImageFile
import metrics
from torch.autograd import Variable,Function
ImageFile.LOAD_TRUNCATED_IMAGES = True
from model import EmbeddingNet,ClassificationNet
from tensorboardX import SummaryWriter

def test_triplet():
    parser = argparse.ArgumentParser(
        description='Face recognition using triplet loss.')
    parser.add_argument('--CVDs', type=str, default='1,2', metavar='CUDA_VISIBLE_DEVICES',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--server', type=int, default=82, metavar='T',
                        help='which server is being used')
    parser.add_argument('--train-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/train2', metavar='dir',
                        help='path of train set.')
    parser.add_argument('--test-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/cifar100/test2', metavar='dir',
                        help='path of train set.')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--embedding-size', type=int, default=128, metavar='N',
                        help='embedding size of model (default: 256)')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='classes number of dataset')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, default='resnet34', metavar='M',
                        help='model name (default: resnet50)')
    parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                        help='Dropout probability (default: 0.2)')
    parser.add_argument('--check-path', type=str, default='/home/zili/memory/FaceRecognition-master/checkpoints2', metavar='C',
                        help='Checkpoint path')
    parser.add_argument('--pretrained', type=bool, default=False,metavar='R',
                        help='whether model is pretrained.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs
    if args.server == 31:
        args.train_set  = '/share/zili/code/triplet/data/mnist/train'
        args.test_set   = '/share/zili/code/triplet/data/test_class'
        args.check_path = '/share/zili/code/triplet/checkpoints2'
    if args.server == 16:
        args.train_set  = '/data0/zili/code/triplet/data/cifar100/train2'
        args.test_set   = '/data0/zili/code/triplet/data/cifar100/test2'
        args.check_path = '/data0/zili/code/triplet/checkpoints2'
    if args.server == 17:
        args.train_set  = '/data/jiaxin/zili/data/cifar100/train2'
        args.test_set   = '/data/jiaxin/zili/data/cifar100/test'
        args.check_path = '/data/jiaxin/zili/checkpoints2'

    now_time = str(datetime.datetime.now())
    args.check_path = os.path.join(args.check_path, now_time)

    if not os.path.exists(args.check_path):
        os.mkdir(args.check_path)

    shutil.copy('crossentropyloss.py', args.check_path)
    # os.path.join(args.check_path)
    f = open(args.check_path + os.path.sep + now_time + 'CrossEntropy.txt', 'w+')

    print('Loading model...')
    # model = FaceModelForCls(model_name=args.model_name,
    #                         num_classes=args.num_classes,
    #                         pretrained=args.pretrained)
    # model = model_.ResNet34(False, num_classes=args.num_classes)
    embedding_net = EmbeddingNet()
    model = ClassificationNet(embedding_net, 10)
    f.write("     model: {}".format(model))
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    transform = transforms.Compose([
        # transforms.Resize(32),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.4914, 0.4822, 0.4465]),
            std=np.array([0.2023, 0.1994, 0.2010])),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    print('Loading data...')
    train_set = torchvision.datasets.ImageFolder(root=args.train_set, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=20)

    test_set = torchvision.datasets.ImageFolder(root=args.test_set,transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    weight = torch.FloatTensor([1.,1.,1.,1.,1.,1.,1,10.,10.,10.]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(),lr=args.lr)

    writer = SummaryWriter()
    def train(epoch):
        model.train()
        total_loss, correct = 0.0, 0
        fea, l = torch.zeros(0), torch.zeros(0)
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = output.max(1)
            fea = torch.cat((fea, output.data.cpu()))
            l = torch.cat((l, target.data.cpu().float()))

            correct += predicted.eq(target).sum().item()
            writer.add_scalar('/Loss', loss.item(), epoch * len(train_loader) + batch_idx)

            if (batch_idx+1) % args.log_interval == 0:
                context = 'Train Epoch: {} [{}/{} ({:.0f}%)], Average loss: {:.6f}'.format(
                          epoch, fea.size()[0], len(train_loader.dataset),
                          100.0 * batch_idx / len(train_loader), total_loss / (batch_idx+1))
                print(context)
                f.write(context + '\r\n')

        writer.add_embedding(mat=fea, metadata=l, global_step=epoch)
        context = 'Train Epoch: {} [{}/{} ({:.0f}%)], Average loss: {:.4f}'.format(
            epoch, len(train_loader.dataset), len(train_loader.dataset),
            100.0 * len(train_loader) / len(train_loader), total_loss / len(train_loader))
        print(context)
        f.write(context + '\r\n')

        context = 'Train set:  Accuracy: {}/{} ({:.3f}%)\n'.format(
             correct, len(train_loader.dataset),
            100. * float(correct) / len(train_loader.dataset))
        print(context)
        f.write(context+'\r\n')


    def test(epoch):
        model.eval()
        test_loss, correct = 0, 0
        target_arr, predict_arr = [], []
        data1 = [0] * args.num_classes
        data2 = [0] * args.num_classes
        data3 = [0] * args.num_classes
        with torch.no_grad():
            for i , (data, target) in enumerate(test_loader):

                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output = model(data)
                test_loss += criterion(output, target)
                _, pred = output.data.max(1)

                for i in range(0, target.size(0)):
                    data1[target[i]] += 1
                    data3[pred[i]] += 1
                    if target[i] == pred[i]:
                        data2[target[i]] += 1

                batch_correct = pred.eq(target).sum().item()
                correct += batch_correct

                predict_arr.append(pred.cpu().numpy())
                target_arr.append(target.data.cpu().numpy())
                writer.add_scalar('/Acc', 100 * float(batch_correct) / data.size(0), epoch * len(test_loader) + i)

            cm_path = args.check_path + '/' + str(epoch) + '_confusematrix'
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

            test_loss /= len(test_loader)
            context = 'Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / float(len(test_loader.dataset)))
            print(context)
            f.write(context + '\r\n')



    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print('start training')
    for epoch in range(args.epochs):
        if (epoch ) % 2 == 0:
            print(" server: {}".format(args.server))
            print(" train-size: {}".format(args.train_set))
            print(" embedding-size: {}".format(args.embedding_size))
            print(" model: {}".format(args.model_name))
            print(" dropout: {}".format(args.dropout_p))
            print(" num_train_set: {}".format(train_set.__len__()))
            print(" check_path: {}".format(args.check_path))
            print(" learing_rate: {}".format(args.lr))
            print(" batch_size: {}".format(args.batch_size))
            print(" pretrained: {}".format(args.pretrained))
            print(" optimizer: {}".format(optimizer))
            f.write(" server: {}".format(args.server) + '\r\n')
            f.write(" train-size: {}".format(args.train_set) + '\r\n')
            f.write(" embedding-size: {}".format(args.embedding_size) + '\r\n')
            f.write(" model: {}".format(args.model_name) + '\r\n')
            f.write(" dropout: {}".format(args.dropout_p) + '\r\n')
            f.write(" num_train_set: {}".format(train_set.__len__()) + '\r\n')
            f.write(" check_path: {}".format(args.check_path) + '\r\n')
            f.write(" learing_rate: {}".format(args.lr) + '\r\n')
            f.write(" batch_size: {}".format(args.batch_size) + '\r\n')
            f.write(" pretrained: {}".format(args.pretrained) + '\r\n')
            f.write(" optimizer: {}".format(optimizer) + '\r\n')
        train(epoch)
        test(epoch)
        if (epoch + 1) % 20 == 0 :
            args.lr = args.lr / 3
            update_lr(optimizer, args.lr)
        if epoch > 30 and epoch < 80:
            torch.save(model, args.check_path+ os.path.sep + 'epoch' + str(epoch)+'.pth')
        f.write('\r\n')
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    f.close()
    # torch.save(model, args.check_path + os.path.sep  +'CrossEntropy.pth')


if __name__ == '__main__':
    test_triplet()

