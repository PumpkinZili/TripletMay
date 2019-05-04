import argparse
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import datetime
from torch.backends import cudnn
import numpy as np
from PIL import ImageFile
from torch.autograd import Variable,Function
ImageFile.LOAD_TRUNCATED_IMAGES = True
from model import FaceModelForCls
from tensorboardX import SummaryWriter

def test_triplet():
    parser = argparse.ArgumentParser(
        description='Face recognition using triplet loss.')
    parser.add_argument('--CVDs', type=str, default='3,2', metavar='CUDA_VISIBLE_DEVICES',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--train-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/mnist/train', metavar='T',
                        help='path of train set.')
    parser.add_argument('--test-set', type=str, default='/home/zili/memory/FaceRecognition-master/data/mnist/test', metavar='T',
                        help='path of train set.')
    parser.add_argument('--batch-size', type=int, default=384, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=384, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--embedding-size', type=int, default=256, metavar='N',
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
    parser.add_argument('--dropout-p', type=float, default=0.5, metavar='D',
                        help='Dropout probability (default: 0.2)')
    parser.add_argument('--check-path', type=str, default='checkpoints2', metavar='C',
                        help='Checkpoint path')
    parser.add_argument('--is-pretrained', type=bool, default=False,metavar='R',
                        help='whether model is pretrained.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs

    output1 = str(datetime.datetime.now())
    f = open(args.check_path + os.path.sep + output1 + 'CrossEntropy.txt', 'w+')

    print('Loading model...')
    model = FaceModelForCls(embedding_size=args.embedding_size,
                            num_classes=args.num_classes,
                            pretrained=args.is_pretrained)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True


    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])


    print('Loading data...')
    train_set = torchvision.datasets.ImageFolder(root=args.train_set, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root=args.test_set,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.test_batch_size, shuffle = True)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(),lr=args.lr)

    writer = SummaryWriter()
    def train(epoch):
        model.train()
        total_loss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, predicted = output.max(1)

            correct += predicted.eq(target).sum().item()

            writer.add_scalar('/Loss', loss.item(), epoch * len(train_loader) + batch_idx)

            if (batch_idx+1) % args.log_interval == 0:
                context = 'Train Epoch: {} [{}/{} ({:.0f}%)], Average loss: {:.6f}'.format(
                          epoch, batch_idx * len(data), len(train_loader.dataset),
                          100.0 * batch_idx / len(train_loader), total_loss / (batch_idx+1))
                print(context)
                f.write(context + '\r\n')
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
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i , (data, target) in enumerate(test_loader):
                # print(i)
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output = model(data)

                test_loss += criterion(output, target)
                _, pred = output.data.max(1)
                batch_correct = pred.eq(target).sum().item()
                correct += batch_correct

                writer.add_scalar('/Acc', 100 * float(batch_correct) / data.size(0), epoch * len(test_loader) + i)

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
        if (epoch ) % 5 == 0:
            print("            embedding-size: {}".format(args.embedding_size))
            print("            model: {}".format(args.model_name))
            print("            dropout: {}".format(args.dropout_p))
            print("            num_train_set: {}".format(train_set.__len__()))
            print("            check_path: {}".format(args.check_path))
            print("            learing_rate: {}".format(args.lr))
            print("            batch_size: {}".format(args.batch_size))
            print("            is_pretrained: {}".format(args.is_pretrained))
            print("            optimizer: {}".format(optimizer))
            f.write("            embedding-size: {}".format(args.embedding_size) + '\r\n')
            f.write("            model: {}".format(args.model_name) + '\r\n')
            f.write("            dropout: {}".format(args.dropout_p) + '\r\n')
            f.write("            num_train_set: {}".format(train_set.__len__()) + '\r\n')
            f.write("            check_path: {}".format(args.check_path) + '\r\n')
            f.write("            learing_rate: {}".format(args.lr) + '\r\n')
            f.write("            batch_size: {}".format(args.batch_size) + '\r\n')
            f.write("            is_pretrained: {}".format(args.is_pretrained) + '\r\n')
            f.write("            optimizer: {}".format(optimizer) + '\r\n')
        train(epoch)
        test(epoch)
        if (epoch + 1) % 10 == 0 and epoch < 53:
            args.lr = args.lr / 3
            update_lr(optimizer, args.lr)

        f.write('\r\n')
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    f.close()
    torch.save(model, args.check_path + os.path.sep + output1 +'_CrossEntropy.pth')


if __name__ == '__main__':
    test_triplet()

