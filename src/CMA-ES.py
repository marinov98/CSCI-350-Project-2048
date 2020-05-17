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


def generate_random_weights_arr(start, end, variables=4, samples=100):
    """ Creates a variables x samples matrix of random numbers based on range """
    weight_combinations = [None] * variables
    for variable in range(variables):
        weight_combinations[variable] = np.random.uniform(start, end, samples)

    means = find_means_of_weights(weight_combinations)

    return weight_combinations, means


def generate_covariance_matrix(data, population=False):
    """ Creates either a population or sample covariance matrix based on the data """
    return np.cov(data, bias=population)


def visualize_covariance_matrix(cov_matrix):
    """ Creates Covariance matrix from weights and their means """
    sn.heatmap(cov_matrix, annot=True, fmt='g')
    plt.show()


def generate_samples(covariance_matrix, means, samples=10):
    """ Generates samples from guassian normal distribution """
    return np.random.Generator.multivariate_normal(covariance_matrix, means, size=samples)
