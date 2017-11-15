#!/usr/bin/env python
import numpy as np


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
                            
        for series in np.arange(self.F):
            for time in np.arange(self.T):
                for cluster in np.arange(self.n_components):
                    self.__p_cluster_sample[cluster][time][series] =\
                        self.__get_p_cluster_sample(cluster, self.X[series][time])
        
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
    
    def __get_p_cluster_sample(self, cluster, x):
        '''
            Conditional population-weight-estimated probability(cluster|sample)
        '''
        return self.cluster_weights[cluster] * self.__p_sample_cluster(x, cluster) /\
            np.sum([self.cluster_weights[i] * self.__p_sample_cluster(x, i) for i in np.arange(self.n_components)])
        
    
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