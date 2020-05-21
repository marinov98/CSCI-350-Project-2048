'''
Final AI Project
Authors: Owen Okunhardt, Aryan Bhatt, Marin Marinov
'''

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


def find_means_of_weights(weight_combinations):
    """ Find means of each weight """
    return [np.mean(heur_weight) for heur_weight in weight_combinations]


def generate_data(start, end, variables=4, samples=100):
    """ Creates a variables x samples matrix of random numbers based on range """
    weight_combinations = [None] * variables
    for variable in range(variables):
        weight_combinations[variable] = np.random.randint(start, end, samples)

    means = find_means_of_weights(weight_combinations)

    return weight_combinations, means


def visualize_covariance_matrix(cov_matrix):
    """ Displays a visualization of a covariance matrix """
    sn.heatmap(cov_matrix, annot=True, fmt='g')
    plt.show()


def get_multivariate_std(data1, mean1, data2=[], mean2=0, same=False):
    """ computes standard deviaton for one or more variables """
    summation = 0
    # for diagonals
    if same:
        for sample in data1:
            summation += ((sample - mean1) * (sample - mean1))
    else:
        for (var1, var2) in zip(data1, data2):
            summation += ((var1 - mean1) * (var2 - mean2))

    summation /= len(data1)
    return summation


def generate_covariance_matrix4(data, means):
    """ Creates a 4x4 covariance matrix based on the data """
    cov_matrix = [[0 for i in range(4)] for j in range(4)]

    # assumes each row in the data is a variable
    x = data[0]
    y = data[1]
    z = data[2]
    w = data[3]

    x_mean = means[0]
    y_mean = means[1]
    z_mean = means[2]
    w_mean = means[3]

    cov_matrix[0][0] = get_multivariate_std(x, x_mean, same=True)
    cov_matrix[0][1] = get_multivariate_std(x, x_mean, y, y_mean)
    cov_matrix[0][2] = get_multivariate_std(x, x_mean, z, z_mean)
    cov_matrix[0][3] = get_multivariate_std(x, x_mean, w, w_mean)
    cov_matrix[1][0] = cov_matrix[0][1]
    cov_matrix[1][1] = get_multivariate_std(y, y_mean, same=True)
    cov_matrix[1][2] = get_multivariate_std(y, y_mean, z, z_mean)
    cov_matrix[1][3] = get_multivariate_std(y, y_mean, w, w_mean)
    cov_matrix[2][0] = cov_matrix[0][2]
    cov_matrix[2][1] = cov_matrix[1][2]
    cov_matrix[2][2] = get_multivariate_std(z, z_mean, same=True)
    cov_matrix[2][3] = get_multivariate_std(z, z_mean, w, w_mean)
    cov_matrix[3][0] = cov_matrix[0][3]
    cov_matrix[3][1] = cov_matrix[1][3]
    cov_matrix[3][2] = cov_matrix[2][3]
    cov_matrix[3][3] = get_multivariate_std(w, w_mean, same=True)

    return cov_matrix


def generate_next_generation_data(map, means, population=False, best_samples=10):
    """ Greates new data for next generation and their means """
    new_gen_data = []

    # get all weight combinations into an array
    for key, val in sorted(map.items()):
        new_gen_data.append(val)

    # get only the best
    new_gen_data_best = new_gen_data[-1 * best_samples:]

    # Create transposed version for easier means and covariance computation
    new_gen_data_transposed = (np.array(new_gen_data_best)).transpose()

    new_gen_data_means = find_means_of_weights(new_gen_data_transposed)

    # recreate covariance matrix with old mean
    new_gen_cov = generate_covariance_matrix4(new_gen_data_transposed, means)

    return new_gen_cov, new_gen_data_means


def generate_normal_distribution(data=[], means=[], cov_matrix=None, population=False, samples=10):
    """ Finds covariance matrix and returns multivarial normal distribution """
    covariance_matrix = np.cov(
        data, bias=population) if not cov_matrix else cov_matrix
    return np.random.default_rng().multivariate_normal(means, covariance_matrix, size=samples)
