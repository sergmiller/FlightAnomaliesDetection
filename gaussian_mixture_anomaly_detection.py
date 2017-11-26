#!/usr/bin/env python
import numpy as np
import pandas as pd

from tqdm import tqdm


class GaussianMixtureInTimeAnomalyDetector:
    '''
        ClusterAD-DataSample anomaly-detection method implementation
    '''
    def __init__(self,
                    n_components=35,
                    tol=1e-6,
                    covariance_type='diag',
                    init_params='kmeans',
                    random_state=None,
                ):
        '''
            Constructor accepts some args for sklearn.mixture.GaussianMixture inside.
            Default params are choosen as the most appropriate for flight-anomaly-detection problem
            according the original article.
        '''

        self.n_components = n_components
        self.tol = tol
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.random_state = random_state

        self.eps = 1e-12  # feature-normalization constant

    def fit(self, X):
        '''
            X must contains F objects time series with length T vectors-features with size N each
            i. e. X.shape is (F, N, M)
        '''
        from sklearn.mixture import GaussianMixture
        from numpy.linalg import norm
        from copy import deepcopy

        X = np.array(X)

        assert len(X.shape) == 3
        self.F, self.T, self.N = X.shape

        # prepare data for fitting
        X = X.reshape(self.F * self.T, self.N)

        self.data_mean = np.mean(X, axis=0)
        self.data_std = np.std(X, axis=0) + self.eps

        X = self._normalize(X)

        gm = GaussianMixture(
            n_components=self.n_components,
            tol=self.tol,
            covariance_type=self.covariance_type,
            init_params=self.init_params,
            random_state=self.random_state,
                            )

        gm.fit(X)

        self.X = X.reshape(self.F, self.T, self.N)

        self.cluster_weights = gm.weights_
        self.cluster_means = gm.means_
        self.cluster_covariances = gm.covariances_

        print('Start probabilities memorization')

        self.__memorize_probs()

        return self.__evaluate_log_likelihood(self.X)

    def predict(self, X):
        '''
            Calculate log-likelihood for each one-time-slice sample,

            dim(X) = 3, X.shape[1] must be equal time length(shape[1])

            Normalize samples by train data and call evaluation log likelihood method
        '''
        X = np.array(X)
        assert len(X.shape) == 3
        assert X.shape[1] == self.T
        X = self.__normalize(X)

        return self._evaluate_log_likelihood(X)


    def __memorize_probs(self):
        # memorization all P(cluster|sample)
        self.__p_cluster_sample = np.zeros((self.n_components, self.T, self.F))

        for series in tqdm(np.arange(self.F), position=0):
            for time in np.arange(self.T):
                probs = [self.cluster_weights[i] * self.__p_sample_cluster(self.X[series][time], i) \
           			for i in np.arange(self.n_components)]
                norma = np.sum(probs)

                for cluster in np.arange(self.n_components):
                    self.__p_cluster_sample[cluster][time][series] = probs[cluster] / norma

        # memorization all P(cluster|time)
        self.__p_cluster_time = np.zeros((self.n_components, self.T))

        for time in np.arange(self.T):
            for cluster in np.arange(self.n_components):
                self.__p_cluster_time[cluster][time] = self.__get_p_cluster_time(cluster, time)


    def _normalize(self, X):
        return (X - self.data_mean) / self.data_std

    def __diag_gauss_pdf(self, x, mean, cov):
        '''
            Custom calculation gaussian density in case if covariance is diagonal matrix

            cov is array of covariance matrix diagonal elements
        '''
        delta = np.array(x) - np.array(mean)
        inv = 1 / (np.array(cov) + self.eps)
        logp = -0.5 * ((delta.dot(inv * delta)) + np.log(np.prod(cov) + self.eps) + self.N * np.log(2 * np.pi))

        return np.exp(logp)

    def __p_sample_cluster(self, x, cluster):
        '''
            Conditional likelihood(sample|cluster)
        '''
        return self.__diag_gauss_pdf(x,
                                    self.cluster_means[cluster],
                                    self.cluster_covariances[cluster])


    def __get_p_cluster_time(self, cluster, t):
        clusters_probs = [np.sum(self.__p_cluster_sample[i][t]) for i in np.arange(self.n_components)]

        return clusters_probs[cluster] / np.sum(clusters_probs)


    def __evaluate_sample_in_time(self, x, t):
        '''
            Evaluation in population log likelihood for time-slice normalized sample x_t.

            x is M-dimentional one-time-slice sample

            t is order(time) of sample x in time series
            t must be in [0, 1, ... N)
        '''

        return np.log(np.sum([self.__p_sample_cluster(x, cluster) * self.__p_cluster_time[cluster][t] \
                         for cluster in np.arange(self.n_components)]))


    def __evaluate_log_likelihood(self, X):
        log_likelihood = np.zeros(X.shape[:2])
        for f in np.arange(X.shape[0]):
            for t in np.arange(X.shape[1]):
                log_likelihood[f][t] = self.__evaluate_sample_in_time(X[f][t], t)

        return log_likelihood


    def smoothed_sample_anomalies(self, scores, halflife=2):
        '''
            extract exponential weighted sample likelihoode
        '''
        frames = [pd.DataFrame(series) for series in scores]
        return np.array([np.array(pd.ewma(series, halflife)).reshape(-1) for series in frames])


    def find_anomalies(self, scores, strategy='sample', anomaly_top=0.01, log_likelihood_threshold=None):
        '''
            extract abnormal samples

            Args:

                scores - log_likelihoods for each one-time-slice sample,

                dim(X) = 2, X.shape[1] must be equal time length(shape[1])

                strategy(default: 'sample') - way of results representation:
                    'sample' - find abnormal one-time-slice samples
                    'series' - find abnormal series (anomality of each series is sum of samples log likelihoods)

                log_likelihood_threshold is used as anomaly-data upper bound if it's specified, else
                anomaly_top will be used and converted to population log_likelihood_threshold

            Return:
                sorted list of anomalies - [(log_likelihood, (series, [time])),]

                log_likelihood_threshold - upper bound of abnormal samples
        '''
        scores = np.array(scores)

        assert len(scores.shape) == 2
        assert scores.shape[1] == self.T
        assert anomaly_top > 0 or log_likelihood_threshold is not None

        if strategy == 'sample':
            serialized_scores = [(scores[series][time], (series, time))
                                 for series in np.arange(scores.shape[0]) for time in np.arange(scores.shape[1])]

        elif strategy == 'series':
            serialized_scores = [(np.sum(scores[series]), (series))
                                 for series in np.arange(scores.shape[0])]
        else:
            raise  ValueError("strategy must be in {}".format(['sample', 'series']))

        serialized_scores.sort()

        sorted_scores = np.array([s[0] for s in serialized_scores])
        size = len(sorted_scores)

        if log_likelihood_threshold is None:
            bound = int(size * anomaly_top)
            bound = min(size, bound + 1)
            log_likelihood_threshold = sorted_scores[max(0, bound - 1)]
        else:
            bound = np.argmax(sorted_scores > log_likelihood_threshold)
            if bound == 0 and sorted_scores[0] <= log_likelihood_threshold:
                bound = len(sorted_scores)

        return serialized_scores[:bound], log_likelihood_threshold


def extract_anomaly_target(frame, frame_period, halflife,
                            horizont, n_components=35, top=0.01):
    assert len(frame.shape) == 2
    assert isinstance(frame, pd.DataFrame)
    data = np.array(frame).reshape(-1, frame_period, frame.shape[1])
    detector = GaussianMixtureInTimeAnomalyDetector(n_components=n_components, random_state=1)
    # scores  - лограифмическое правдоподобие нормальности для каждого сэмпла
    scores = detector.fit(data)
    smoothed_scores = detector.smoothed_sample_anomalies(scores, halflife)
    anomalies, treshold = detector.find_anomalies(scores, anomaly_top=top)
    anomaly_indexes = [t[1][0] * frame_period + t[1][1] for t in anomalies]
    all_anomalies = set()
    for a in anomaly_indexes:
        for l in np.arange(horizont):
            all_anomalies.add(max(a - l, 0))

    targets = np.zeros(frame.shape[0])

    for a in all_anomalies:
        targets[a] = 1

    return targets
