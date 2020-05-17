import numpy as np

   def generate_random_weights_arr(start, end, iterations):
        """ Creates random array of 4 weights """
        res = []
        for i in range(iterations):
            res.append(np.random.uniform(start, end, 4))

        print(res)

    def create_covariance_matrix(weights_arr, means):
        """ Creates Covariance matrix from weights and their means """
        return "the thing"

    def generate_samples(covariance_matrix, means, samples=10):
        """ Generates samples from guassian normal distribution """
        return "the other thing"

generate_random_weights_arr(0.1, 500, 100)
