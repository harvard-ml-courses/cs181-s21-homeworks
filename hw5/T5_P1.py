import time
import numpy as np
import pandas as pd

class EM_Geometric(object):
    def __init__(self, num_clusters, min_step=1e-4, probs=None, priors=None):
        '''
        Initialize the EM algorithm based on given parameters

        Parameters
        ----------
        num_clusters : int
            The number of labels the EM algorithm should form clusters for
        min_step : int
            The minimum steps required for termination of the algorithm
        probs : np.array
            The initial parameter p for each label (geometric distribution)
        priors : np.array
            The initial mixing proportion of each label (aka theta)
        '''
        self.num_clusters = num_clusters
        self.min_step = min_step

        if probs is None:
            self.probs = np.random.random(self.num_clusters)
        else:
            self.probs = probs

        if priors is None:
            self.pis = np.ones(self.num_clusters) / self.num_clusters
        else:
            self.pis = priors

    def cluster(self, data, max_iters=50):
        '''
        Train on the given data to form clusters for the labels

        Parameters
        ----------
        data : pd.DataFrame
            The feature values of the data set
        '''
        self.data = data
        start_time = time.time()

        # Initialize the soft cluster assignments q for each data point to each label
        self.q = np.asmatrix(np.empty((len(self.data), self.num_clusters), dtype=float))

        num_iters = 0

        # Current and previous log-likelihoods
        curr_ll = 0
        prev_ll = 0

        # While termination conditions have not been met, run the algorithm
        while num_iters <= 1 or (curr_ll - prev_ll > self.min_step and num_iters < max_iters):
            # Perform the expectation step
            self.e_step()
            # Perform the maximization step
            self.m_step()

            num_iters += 1

            # Calculate and compare log-likelihoods
            prev_ll = curr_ll
            curr_ll = self.loglikelihood()
            print(f'Iteration {num_iters}, log-likelihood: {curr_ll}')

        total_time = time.time() - start_time
        print(f'Complete, time elapsed: {total_time} seconds')

        # Sort the clusters and re-arrange the soft labels
        i = np.argsort(self.probs)
        self.probs = np.sort(self.probs)
        self.q = self.q[:, i]

    def loglikelihood(self):
        '''
        Calculate the current log-likelihood given data and soft assignments

        Returns
        -------
        ll : float
            The log-likelihood of current set of labels (according to a geometric distribution)
        '''
        # TODO: Implement the log-likelihood function
        pass

    def e_step(self):
        '''
        Run the expectation step, calculating the soft assignments based on current parameter estimates
        '''
        # TODO: Implement the expectation step
        pass

    def m_step(self):
        '''
        Run the maximization step, calculating parameter estimates based on current soft assignments
        '''
        # TODO: Implement the maximization step
        pass

    def get_labels(self):
        '''
        Return the label with highest probability among the soft assignments for each data point
        '''
        return np.array([np.argmax(q) for q in np.array(self.q)])

def generate_geom_data(num_data, cluster_probs):
    '''
    Generate an equal number of data points for each cluster, based on geometric distribution
    '''
    data = np.array([])
    labels = np.array([], dtype=int)
    for i, prob in enumerate(cluster_probs):
        data = np.concatenate((data, np.random.geometric(p=prob, size=num_data)))
        labels = np.concatenate((labels, np.full(num_data, i, dtype=int)))
    return data, labels

def main():
    # TODO: Toggle these between 10 / 1000 and [0.1, 0.5, 0.9] / [0.1, 0.2, 0.9]
    num_data = 10
    cluster_probs = [0.1, 0.5, 0.9]

    # Do not edit the below code, it is to help you run the algorithm
    # -------------------------------------------------------------------------------
    data, labels = generate_geom_data(num_data=num_data, cluster_probs=cluster_probs)

    em_algo = EM_Geometric(num_clusters=3)
    em_algo.cluster(data)
    print(f'Final probabilities: {em_algo.probs}')

    accuracy = np.equal(em_algo.get_labels(), labels).sum() / len(labels)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()