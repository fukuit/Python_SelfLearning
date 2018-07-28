'''
PyTorch MNIST sample
'''

import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from net import Net


def parser():
    '''
    argument
    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', '-l', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()
    return args


def main(device):
    '''
    main
    '''
    args = parser()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])

    trainset = MNIST(root='./data',
                     train=True,
                     download=True,
                     transform=transform)
    testset = MNIST(root='./data',
                    train=False,
                    download=True,
                    transform=transform)

    trainloader = DataLoader(trainset,
                             batch_size=100,
                             shuffle=True,
                             num_workers=2)
    testloader = DataLoader(testset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=2)

    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    # model
    net = Net()
    net.to(device)

    # define loss function and optimier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr, momentum=0.95, nesterov=True)

    # train
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 40 == 39:
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch+1, i+1, running_loss/40))
                running_loss = 0.0
    print('Finished Training')

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))

if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device)
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))
