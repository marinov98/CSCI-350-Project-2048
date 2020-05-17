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


def generate_next_generation_data(data, means, population=False, best_samples=10):
    """ Greates new data for next generation and their means """
    new_gen_data = [] # perhaps some kind of preprocessing

    new_gen_data_mean = [np.mean(heur_weight) for heur_weight in new_gen_data]
    # grab best samples and recalculate means
    # needs more work

    return new_gen_data, new_gen_data


def generate_normal_distribution(data, means, cov_matrix=None, population=False, samples=10):
    """ Finds covariance matrix and returns multivarial normal distribution """
    covariance_matrix = np.cov(
        data, bias=population) if not cov_matrix else cov_matrix
    return np.random.Generator.multivariate_normal(covariance_matrix, means, size=samples)
