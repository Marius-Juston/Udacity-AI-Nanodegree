import argparse
import json

# import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from final_project.helper import create_classifier, retrieve_pre_trained_model, create_optimizer


def get_input_args():
    """
       Retrieves and parses the command line arguments created and defined using
       the argparse module. This function returns these arguments as an
       ArgumentParser object.
        5 command line arguements are created:
            input - the image to predict
            checkpoint - the saved model checkpoint
            category_names - the translation from index to name file location
            top_k - the top k results from the prediction
            gpu - to use the gpu or not
       Parameters:
        None - simply using argparse module to create & store command line arguments
       Returns:
        parse_args() -data structure that stores the command line arguments object
       """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
                        help='Image to predict class')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The model checkpoint location')

    parser.add_argument('--category_names', type=str, default="cat_to_name.json",
                        help='The model checkpoint location')

    parser.add_argument('--top_k', type=int, default=5,
                        help='Return k best answer from the classifier')

    parser.add_argument("--gpu", action="store_true", default=False)

    return parser.parse_args()


def load_checkpoint(file_name):
    parameters = torch.load(file_name)

    # parameter_dict = {
    #     "classifier_state_dict": model.classifier.state_dict(),
    #     "classifier_outputs": outputs,
    #     "classifier_hidden": hidden,
    #     "classifier_dropout_percentage": dropout_percentage,
    #     "lr": lr,
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "epoch": epoch,
    #     "step": step,
    #     "optimizer": optimizer_name,
    #     "model": model_name,
    #     "class_to_idx": train_dataset.class_to_idx,
    #     "idx_to_class": dict((v, k) for k, v in train_dataset.class_to_idx.items())
    # }

    model = retrieve_pre_trained_model(parameters["model"])

    classifier = create_classifier(model.classifier.in_features,
                                   parameters["classifier_hidden"],
                                   parameters["classifier_outputs"],
                                   parameters["classifier_dropout_percentage"])
    classifier.load_state_dict(parameters["classifier_state_dict"])

    model.classifier = classifier
    model.class_to_idx = parameters["class_to_idx"]
    model.idx_to_class = parameters["idx_to_class"]

    model.load_state_dict(parameters["model_state_dict"])

    optimizer = create_optimizer(parameters["optimizer"], model, parameters["lr"])
    optimizer.load_state_dict(parameters["optimizer_state_dict"])

    epoch, step = parameters["epoch"], parameters["step"]

    return model, optimizer, epoch, step


def process_image(img):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """

    img = Image.open(img)
    img.thumbnail((256, 256))

    width, height = img.size  # Get dimensions

    new_width = 224
    new_height = 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    img = img.crop((left, top, right, bottom))

    np_image = np.array(img)
    img = np_image / 255

    mean = np.array([0.485, 0.456, 0.406])
    stdv = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / stdv

    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img)


def predict(image, model, topk=5, gpu=False):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    image = np.array([image.numpy()], dtype=np.float64)
    image = torch.from_numpy(image).float()
    image = Variable(image)

    model.eval()
    if gpu:
        model.cuda()
        image = image.cuda()

    outputs = model.forward(image)

    probabilities = torch.exp(outputs).data
    top_results = probabilities.topk(topk)

    ps = list(top_results[0].cpu().numpy()[0])

    classes = [model.idx_to_class[idx] for idx in top_results[1].cpu().numpy()[0]]

    return ps, classes


# def imshow(image, ax=None, title=None):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.numpy().transpose((1, 2, 0))
#
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
#
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
#
#     ax.imshow(image)
#     ax.set_title(title)
#
#     return ax


# def show_prediction(image_dir, processed_image, class_to_name, classes, probabilities):
#     fig, (image_ax, bar_ax) = plt.subplots(nrows=2)
#
#     imshow(processed_image, ax=image_ax, title=class_to_name[image_dir.split("/")[-2]])
#     image_ax.axis('off')
#
#     y_pos = np.arange(len(classes))
#     bar_ax.barh(y_pos, probabilities[::-1], align='center',
#                 color='blue')
#     bar_ax.set_yticks(y_pos)
#     bar_ax.set_yticklabels(classes[::-1])
#
#     for class_name, ps in zip(classes, probabilities):
#         print(class_name, ps)
#
#     return fig


def main():
    arguments = get_input_args()

    model, optimizer, epoch, step = load_checkpoint(arguments.checkpoint)

    image = arguments.input
    processed_image = process_image(image)

    gpu = arguments.gpu and torch.cuda.is_available()

    ps, classes = predict(processed_image, model, arguments.top_k, gpu)

    with open(arguments.category_names, 'r') as f:
        category_names = json.load(f)

    classes = [category_names[idx] for idx in classes]

    print("--------------------------- PREDICTION --------------------------- ")

    for class_name, ps in zip(classes, ps):
        print(class_name, ps)

    # fig = show_prediction(image, processed_image, category_names, classes, ps)

    # plt.show()


if __name__ == '__main__':
    main()
