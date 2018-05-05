from collections import OrderedDict

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

train_dataset = ImageFolder(train_dir, transform=train_transforms)
test_dataset = ImageFolder(test_dir, transform=train_transforms)
validation_dataset = ImageFolder(valid_dir, transform=validation_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = models.densenet121(pretrained=True)

for parameter in model.parameters():
    parameter.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('d1', nn.Dropout(.1)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
cuda = is_available()

epochs = 3
steps = 0
running_loss = 0
print_every = 40

model.cuda()

model.load_state_dict()

for e in range(epochs):
    # Model in training mode, dropout is on
    model.train()

    for ii, (inputs, labels) in enumerate(train_dataloader):
        # inputs = inputs.resize_(inputs.size()[0], 224 * 224 * 3)
        # print(inputs.size())
        inputs, labels = Variable(inputs), Variable(labels)
        steps += 1

        optimizer.zero_grad()

        # Move input and label tensors to the GPU
        inputs, labels = inputs.cuda(), labels.cuda()

        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]

        if steps % print_every == 0:
            # Model in inference mode, dropout is off
            model.eval()

            accuracy = 0
            test_loss = 0
            for ii, (inputs, labels) in enumerate(valid_dataloader):
                # Move input and label tensors to the GPU
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                output = model.forward(inputs)
                test_loss += criterion(output, labels).data[0]

                ## Calculating the accuracy
                # Model's output is log-softmax, take exponential to get the probabilities
                ps = torch.exp(output).data
                # Class with highest probability is our predicted class, compare with true label
                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.FloatTensor()).mean()

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(valid_dataloader)),
                  "Test Accuracy: {:.3f}".format(accuracy / len(valid_dataloader)))

            running_loss = 0

            # Make sure dropout is on for training
            model.train()
