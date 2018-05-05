import numpy as np

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W) + b)[0])


# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate=0.01):
    # Fill in code
    # p_pred = np.matmul(X, W) + b
    # p_pred[p_pred >= 0] = 1
    # p_pred[p_pred < 0] = 0

    for i in range(X.shape[0]):
        x = X[i]
        pred = prediction(x, W, b)
        x = x.reshape(-1, 1)

        if pred != y[i]:
            if pred == 0:
                W += x * learn_rate
                b += learn_rate
            else:
                W -= x * learn_rate
                b -= learn_rate

    return W, b


# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
    return boundary_lines


X = np.array([[0.78051, -0.063669],
              [0.28774, 0.29139],
              [0.40714, 0.17878],
              [0.2923, 0.4217],
              [0.50922, 0.35256],
              [0.27785, 0.10802],
              [0.27527, 0.33223],
              [0.43999, 0.31245],
              [0.33557, 0.42984],
              [0.23448, 0.24986],
              [0.0084492, 0.13658],
              [0.12419, 0.33595],
              [0.25644, 0.42624],
              [0.4591, 0.40426],
              [0.44547, 0.45117],
              [0.42218, 0.20118],
              [0.49563, 0.21445],
              [0.30848, 0.24306],
              [0.39707, 0.44438],
              [0.32945, 0.39217],
              [0.40739, 0.40271],
              [0.3106, 0.50702],
              [0.49638, 0.45384],
              [0.10073, 0.32053],
              [0.69907, 0.37307],
              [0.29767, 0.69648],
              [0.15099, 0.57341],
              [0.16427, 0.27759],
              [0.33259, 0.055964],
              [0.53741, 0.28637],
              [0.19503, 0.36879],
              [0.40278, 0.035148],
              [0.21296, 0.55169],
              [0.48447, 0.56991],
              [0.25476, 0.34596],
              [0.21726, 0.28641],
              [0.67078, 0.46538],
              [0.3815, 0.4622],
              [0.53838, 0.32774],
              [0.4849, 0.26071],
              [0.37095, 0.38809],
              [0.54527, 0.63911],
              [0.32149, 0.12007],
              [0.42216, 0.61666],
              [0.10194, 0.060408],
              [0.15254, 0.2168],
              [0.45558, 0.43769],
              [0.28488, 0.52142],
              [0.27633, 0.21264],
              [0.39748, 0.31902],
              [0.5533, 1.],
              [0.44274, 0.59205],
              [0.85176, 0.6612],
              [0.60436, 0.86605],
              [0.68243, 0.48301],
              [1., 0.76815],
              [0.72989, 0.8107],
              [0.67377, 0.77975],
              [0.78761, 0.58177],
              [0.71442, 0.7668],
              [0.49379, 0.54226],
              [0.78974, 0.74233],
              [0.67905, 0.60921],
              [0.6642, 0.72519],
              [0.79396, 0.56789],
              [0.70758, 0.76022],
              [0.59421, 0.61857],
              [0.49364, 0.56224],
              [0.77707, 0.35025],
              [0.79785, 0.76921],
              [0.70876, 0.96764],
              [0.69176, 0.60865],
              [0.66408, 0.92075],
              [0.65973, 0.66666],
              [0.64574, 0.56845],
              [0.89639, 0.7085],
              [0.85476, 0.63167],
              [0.62091, 0.80424],
              [0.79057, 0.56108],
              [0.58935, 0.71582],
              [0.56846, 0.7406],
              [0.65912, 0.71548],
              [0.70938, 0.74041],
              [0.59154, 0.62927],
              [0.45829, 0.4641],
              [0.79982, 0.74847],
              [0.60974, 0.54757],
              [0.68127, 0.86985],
              [0.76694, 0.64736],
              [0.69048, 0.83058],
              [0.68122, 0.96541],
              [0.73229, 0.64245],
              [0.76145, 0.60138],
              [0.58985, 0.86955],
              [0.73145, 0.74516],
              [0.77029, 0.7014],
              [0.73156, 0.71782],
              [0.44556, 0.57991],
              [0.85275, 0.85987],
              [0.51912, 0.62359]])

y = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1, 1., 1., 1., 1.,
              1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1, 1., 1., 1., 1.,
              1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0., 0., 0.,
              0., 0., 0., 0.])

trainPerceptronAlgorithm(X, y)
