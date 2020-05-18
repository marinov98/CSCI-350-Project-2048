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
        weight_combinations[variable] = np.random.uniform(start, end, samples)

    means = find_means_of_weights(weight_combinations)

    return weight_combinations, means


def visualize_covariance_matrix(cov_matrix):
    """ Displays a visualization of a covariance matrix """
    sn.heatmap(cov_matrix, annot=True, fmt='g')
    plt.show()


def get_multivariate_std(data, mean, var1, var2, best_samples):
    """ computes standard deviaton for two variables """


def generate_covariance_matrix4(data, means):
    """ Creates a 4x4 covariance matrix based on the data """
    cov_matrix = [[0 for i in range(4)] for j in range(4)]

    return cov_matrix


def generate_next_generation_data(map, means, population=False, best_samples=10):
    """ Greates new data for next generation and their means """
    sorted_data = sorted(map.keys())
    new_gen_data = []

    # get all weight combinations into an array
    for val in sorted_data.values():
        new_gen_data.append(val)

    # get only the best
    new_gen_data_best = new_gen_data[-1 * best_samples:]

    # Create transposed version for easier means and covariance computation
    new_gen_data_transposed = (np.array(new_gen_data_best)).transpose()

    new_gen_data_means = [np.mean(heur_weight)
                          for heur_weight in new_gen_data_transposed]

    # recreate covariance matrix with old mean
    new_gen_cov = generate_covariance_matrix4(new_gen_data_transposed, means)

    return new_gen_cov, new_gen_data_means


def generate_normal_distribution(data=[], means=[], cov_matrix=None, population=False, samples=10):
    """ Finds covariance matrix and returns multivarial normal distribution """
    covariance_matrix = np.cov(
        data, bias=population) if not cov_matrix else cov_matrix
    return np.random.Generator.multivariate_normal(covariance_matrix, means, size=samples)
