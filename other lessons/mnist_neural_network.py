import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


# import helper

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    # Download and load the training data
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
    # plt.show()

    model = Network()
    model.fc1.bias.data.fill_(0)
    model.fc1.weight.data.normal_(std=.01)

    # Grab some data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    images.resize_(64, 1, 28 * 28)

    inputs = Variable(images)

    img_idx = 0
    logits = model.forward(inputs[img_idx, :])

    ps = F.softmax(logits, dim=1)

    print(ps)
