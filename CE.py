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
import metrics
from model import EmbeddingNet,ClassificationNet
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(
    description='Face recognition using triplet loss.')
parser.add_argument('--CVDs', type=str, default='0, 3', metavar='CUDA_VISIBLE_DEVICES',
                    help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--train-set', type=str, default='/data0/weipeng/Data/CIFAR100/train', metavar='T',
                    help='path of train set.')
parser.add_argument('--test-set', type=str, default='/data0/weipeng/Data/CIFAR100/test', metavar='T',
                    help='path of train set.')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='classes number of dataset')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-name', type=str, default='Resnet', metavar='M',
                    help='model name (default: resnet50)')
parser.add_argument('--check-path', type=str, default='checkpoints', metavar='C',
                    help='Checkpoint path')
parser.add_argument('--is-pretrained', type=bool, default=False, metavar='R',
                    help='whether model is pretrained.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.CVDs

def main():
    sys_time = str(datetime.datetime.now())
    if not os.path.exists(args.check_path):
        os.mkdir(args.check_path)
    log_path = args.check_path + os.path.sep + sys_time + 'CE.txt'
    log_file = open(log_path, 'w+')

    print('Loading model......')

    # model = torch.load(log_path)
    embedding_net = EmbeddingNet()
    model = ClassificationNet(embedding_net,10)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])

    print('Loading data...')
    train_set = torchvision.datasets.ImageFolder(root=args.train_set, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root=args.test_set, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)
    classweight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
    classweight = np.array(classweight)
    classweight = torch.from_numpy(classweight).float().cuda()
    criterion = nn.CrossEntropyLoss(weight=classweight)

    # ignored_params = list(map(id, model.fc.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params,
    #                      model.parameters())
    #
    # optimizer = torch.optim.SGD([
    #     {'params': base_params},
    #     {'params': model.fc.parameters(), 'lr': 1e-2}
    # ], lr=1e-3, momentum=0.9)
    optimizer = optim.SGD([
                {'params': model.module.model.parameters()},
                {'params': model.module.fc.parameters(), 'lr': 1e-2}
            ], lr=1e-1, momentum=0.9)

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

            if batch_idx % args.log_interval == 0:
                context = 'Train Epoch: {} [{}/{} ({:.0f}%)], Average loss: {:.6f}'.format(
                          epoch, batch_idx * len(data), len(train_loader.dataset),
                          100.0 * batch_idx / len(train_loader), total_loss / (batch_idx+1))
                print(context)
                log_file.write(context + '\r\n')

        context = 'Train set:  Accuracy: {}/{} ({:.3f}%)\n'.format(
             correct, len(train_loader.dataset),
            100. * float(correct) / len(train_loader.dataset))
        print(context)
        log_file.write(context+'\r\n')

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        target_arr = []
        predict_arr = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output = model(data)

                test_loss += criterion(output, target)
                _, pred = output.data.max(1)
                batch_correct = pred.eq(target).sum().item()
                correct += batch_correct

                predict_arr.append(pred.cpu().numpy())
                target_arr.append(target.data.cpu().numpy())
                writer.add_scalar('/Acc', 100 * float(batch_correct) / data.size(0), epoch * len(test_loader) + batch_idx)

            cm_path = './' + str(epoch)+'_confusematrix'
            cm = metrics.get_confuse_matrix(predict_arr, target_arr)
            np.save(cm_path, cm)
            test_loss /= len(test_loader)
            context = 'Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / float(len(test_loader.dataset)))
            print(context)
            log_file.write(context + '\r\n')

    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print('start training')
    for epoch in range(args.epochs):
        if epoch % 5 == 0:
            print("            model: {}".format(args.model_name))
            print("            num_triplet: {}".format(train_set.__len__()))
            print("            check_path: {}".format(args.check_path))
            print("            learing_rate: {}".format(args.lr))
            print("            batch_size: {}".format(args.batch_size))
            print("            is_pretrained: {}".format(args.is_pretrained))
            print("            optimizer: {}".format(optimizer))
            log_file.write("            model: {}".format(args.model_name) + '\r\n')
            log_file.write("            num_triplet: {}".format(train_set.__len__()) + '\r\n')
            log_file.write("            check_path: {}".format(args.check_path) + '\r\n')
            log_file.write("            learing_rate: {}".format(args.lr) + '\r\n')
            log_file.write("            batch_size: {}".format(args.batch_size) + '\r\n')
            log_file.write("            optimizer: {}".format(optimizer) + '\r\n')
        train(epoch)
        test(epoch)
        if epoch == 50:
            args.lr = args.lr / 5
            update_lr(optimizer, args.lr)
        elif epoch == 100:
            args.lr = args.lr / 5
            update_lr(optimizer, args.lr)
        elif epoch == 150:
            args.lr = args.lr / 5
            update_lr(optimizer, args.lr)
        log_file.write('\r\n')

    model_path = args.check_path + os.path.sep + sys_time + 'model.pkt'
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    log_file.close()
    torch.save(model, model_path)


if __name__ == '__main__':
    main()





