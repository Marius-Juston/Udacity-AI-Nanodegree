from torch import optim, nn
from torchvision import models

from final_project.network import Network


def retrieve_pre_trained_model(model_name):
    model = getattr(models, model_name)(pretrained=True)

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def retrieve_classifier(model, classifier_name):
    return getattr(model, classifier_name)


def create_classifier(inputs_size, hidden_layer, output_size, dropout_percentage):
    return Network(inputs_size, output_size, hidden_layer, dropout_percentage)


def create_optimizer(optimizer_name, model, classifier_name, learning_rate):
    return getattr(optim, optimizer_name)(retrieve_classifier(model, classifier_name).parameters(), learning_rate)


def create_criterion(criterion_name):
    return getattr(nn, criterion_name)()
