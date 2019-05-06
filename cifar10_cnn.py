# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)

    train_size = int(0.8 * len(trainset))
    validation_size = len(trainset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(trainset, [train_size, validation_size])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    validationloader=torch.utils.data.DataLoader(validation_dataset, batch_size=128, shuffle=True, num_workers=2)
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    class Cifar10NetCNN(nn.Module):
        def __init__(self):
            super(Cifar10NetCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            # print(x.size()) #torch.Size([4, 16, 5, 5]) 第一列 batch size 第二列channel
            # print(x.size())
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(3 * 32 * 32, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = x.view(-1, 3 * 32 * 32)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x




    #net = Net()
    net =Cifar10NetCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    file_object = open('accuracy_cifar10_CNN.txt','w')

    for epoch in range(100):
        start = time.clock()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  #print(inputs.size()) #torch.Size([4, 3, 32, 32])
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        #torch.save(net, 'cifar10_fc_{num_epoch}epoch.pkl'.format(num_epoch=epoch+1))
        torch.save(net.state_dict(), 'cifar10_cnn_{num_epoch}epoch.pkl'.format(num_epoch=epoch+1))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validationloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Epoch %d : Accuracy of the network on the validation images: %d %%' % (epoch+1, 100 * correct / total))
        file_object.write('Epoch %d : Accuracy of the network on the validation images: %d %%\n' % (epoch+1, 100 * correct / total))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Epoch %d : Accuracy of the network on the train images: %d %%' % (epoch+1, 100 * correct / total))
        file_object.write(
            'Epoch %d : Accuracy of the network on the train images: %d %%\n' % (epoch+1, 100 * correct / total))
        end = time.clock()
        print('Epoch %d : Running time: %s Seconds' % (epoch+1,end - start))  # Running time: 59 Seconds / each epoch
    file_object.close()
    print('Finished Training')
