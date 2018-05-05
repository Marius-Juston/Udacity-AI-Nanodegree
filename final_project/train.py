import argparse

import torch
from torch import cuda
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from final_project.helper import create_optimizer, create_criterion, create_classifier, retrieve_pre_trained_model, \
    retrieve_classifier


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     11 command line arguements are created:
       save_dir - Path to the mage files(default- 'flowers/')
       arch - CNN model architecture to use for image classification(default densenet121 resnet)
       learning_rate - the learning rate of the Adam optimizer
       hidden_units - the number of neurons for the hidden unit
       epochs - the number of epochs to train the network for
       print_every - how many steps to then validate the current model
       dropout_percentage - the dropout percent of the classifier when training
       output_size - the output size of the classifier
       optimizer - the optimizer to use default Adam optimizer
       criterion - the type of loss function to use default NLLLoss
       gpu - to use the gpu or not
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='flowers/',
                        help='path to the images folder')

    parser.add_argument('--arch', type=str, default="densenet121",
                        help='CNN Model architecture')

    parser.add_argument('--learning_rate', type=float, default=0.00076,
                        help='Learning rate for the Adam optimizer')

    parser.add_argument('--hidden_units', nargs='*', default=[500], type=int,
                        help='The number of nodes for the hidden layer')

    parser.add_argument('--epochs', type=int, default=20,
                        help='The number of epoch for training the network')

    parser.add_argument('--print_every', type=int, default=40,
                        help='The number of steps to complete before validating algorithm')

    parser.add_argument('--dropout_percentage', type=float, default=.1,
                        help="The dropout percentage when training the network")

    parser.add_argument('--output_size', type=int, default=102,
                        help='The number of outputs of the classifier')

    parser.add_argument('--optimizer', type=str, default="Adam",
                        help='The type of optimizer to use')

    parser.add_argument('--criterion', type=str, default="NLLLoss",
                        help='The type of optimizer to use')

    parser.add_argument("--gpu", action="store_true", default=False)

    return parser.parse_args()


def get_datasets(save_dir):
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

    train_dir = save_dir + '/train'
    valid_dir = save_dir + '/valid'
    test_dir = save_dir + '/test'

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    test_dataset = ImageFolder(test_dir, transform=test_transforms)
    validation_dataset = ImageFolder(valid_dir, transform=validation_transforms)

    return train_dataset, test_dataset, validation_dataset


def generate_dataloaders(train_dataset, test_dataset, validation_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader, valid_dataloader


def train_model(model, gpu, epochs, print_every, train_dataloader, valid_dataloader, criterion, optimizer):
    steps = 0
    running_loss = 0

    if gpu:
        model.cuda()

    e = 0

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
            if gpu:
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
                    inputs, labels = Variable(inputs), Variable(labels)

                    if gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

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
                      "Validation Loss: {:.3f}.. ".format(test_loss / len(valid_dataloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(valid_dataloader)))

                running_loss = 0

                # Make sure dropout is on for training
                model.train()

    print("Finished Training")
    return e, steps


def test_model(model, test_dataloader, criterion, gpu):
    model.eval()

    accuracy = 0
    test_loss = 0

    if gpu:
        model.cuda()

    for ii, (inputs, labels) in enumerate(test_dataloader):
        # Move input and label tensors to the GPU
        inputs, labels = Variable(inputs), Variable(labels)
        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output).data
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss / len(test_dataloader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(test_dataloader)))


def save_checkpoint(model, outputs, hidden, lr, optimizer, epoch, step, optimizer_name, model_name, dropout_percentage,
                    train_dataset, classifier_name):
    parameter_dict = {
        "classifier_state_dict": retrieve_classifier(model, classifier_name).state_dict(),
        "classifier_outputs": outputs,
        "classifier_hidden": hidden,
        "classifier_dropout_percentage": dropout_percentage,
        "lr": lr,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "optimizer": optimizer_name,
        "model": model_name,
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": dict((v, k) for k, v in train_dataset.class_to_idx.items())
    }

    torch.save(parameter_dict, "checkpoint.chk")


def get_number_of_inputs(model):
    inp = None
    classifier_name = "classifier"
    if hasattr(model, 'fc'):
        classifier_name = 'fc'
        inp = retrieve_classifier(model, classifier_name)
    elif isinstance(model.classifier, nn.Linear):
        inp = model.classifier
    elif isinstance(model.classifier, nn.Sequential):
        inp = model.classifier[0]
    return inp.in_features, classifier_name


def main():
    arguments = get_input_args()

    train_dataset, test_dataset, validation_dataset = get_datasets(arguments.save_dir)
    train_dataloader, test_dataloader, valid_dataloader = generate_dataloaders(train_dataset, test_dataset,
                                                                               validation_dataset)

    model = retrieve_pre_trained_model(arguments.arch)

    input_size, classifier_name = get_number_of_inputs(model)

    classifier = create_classifier(input_size,
                                   arguments.hidden_units,
                                   arguments.output_size,
                                   arguments.dropout_percentage)

    setattr(model, classifier_name, classifier)
    # model.classifier = classifier

    optimizer = create_optimizer(arguments.optimizer, model, classifier_name, arguments.learning_rate)
    loss = create_criterion(arguments.criterion)

    gpu = arguments.gpu and cuda.is_available()

    print("--------------------------- TRAINING MODEL ---------------------------")

    epoch, step = train_model(model,
                              gpu,
                              arguments.epochs,
                              arguments.print_every,
                              train_dataloader,
                              valid_dataloader,
                              loss,
                              optimizer)

    print("--------------------------- TESTING MODEL ---------------------------")

    test_model(model, test_dataloader, loss, gpu)

    print("--------------------------- SAVING MODEL --------------------------- ")

    save_checkpoint(model, arguments.output_size, arguments.hidden_units, arguments.learning_rate, optimizer, epoch,
                    step, arguments.optimizer, arguments.arch, arguments.dropout_percentage, train_dataset,
                    classifier_name)


if __name__ == '__main__':
    main()
