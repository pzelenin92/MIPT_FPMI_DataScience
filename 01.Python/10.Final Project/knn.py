import numpy as np

class KNN:
    """
    K-neariest-neighbor классификатор с L1 нормой
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                diff_test_train = X[i_test] - self.train_X[i_train]
                abs_diff_test_train= np.abs(diff_test_train)
                sum_abs_diff_test_train = np.sum(abs_diff_test_train)
                dists[i_test,i_train] = sum_abs_diff_test_train
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            diff_test_train = X[i_test] - self.train_X
            abs_diff_test_train = np.abs(diff_test_train)
            sum_abs_diff_test_train = np.sum(abs_diff_test_train,axis=1)
            dists[i_test] = sum_abs_diff_test_train
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!

        #first realisation
        # test_tiled = np.tile(X,num_train)
        # train_flattened = self.train_X.flatten()
        # diff_test_train = test_tiled - train_flattened
        # abs_diff_test_train = np.abs(diff_test_train)
        # reshaped_abs_diff_test_train = abs_diff_test_train.reshape(num_test,num_train,-1)
        # sum_reshaped_abs_diff_test_train = np.sum(reshaped_abs_diff_test_train,axis=2)
        # np.copyto(dists,sum_reshaped_abs_diff_test_train)

        # second realisation
        dists = np.sum(np.abs(X[:,np.newaxis,:]-self.train_X[np.newaxis,:,:]),axis=2)

        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            dists_sorted_indexes = np.argsort(dists[i])
            nearest_neighbours = self.train_y[dists_sorted_indexes[:self.k]]
            pred[i] = nearest_neighbours.max()
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            dists_sorted_indexes = np.argsort(dists[i])
            nearest_neighbours = self.train_y[dists_sorted_indexes[:self.k]]
            pred[i] = nearest_neighbours.max()
        return pred
