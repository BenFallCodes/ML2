import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100, init="kmeans++", n_init=10, tolerance=1e-4):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
            init (str): initialization method, either "kmeans++" or "random"
            n_init (int): number of times the algorithm will be run with different centroid seeds
            tolerance (float): relative tolerance with regards to Frobenius norm of the difference
                               in the cluster centers of two consecutive iterations to declare convergence
        """

        self.K = K
        self.max_iters = max_iters
        self.init_method = init
        self.n_init = n_init
        self.tolerance = tolerance


    def init_centers(self, data):
        """
        Pick K data points from the data as initial cluster centers.
        Supports both random selection and KMeans++.

        Arguments:
            data: array of shape (NxD) where N is the number of data points and D is the number of features.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        if self.init_method == "random":
            random_idx = np.random.permutation(data.shape[0])[:self.K]
            centers = data[random_idx]
            return centers
        elif self.init_method == "kmeans++":
            N, D = data.shape
            centers = np.empty((self.K, D))
            
            # 1. Randomly select the first center
            first_idx = np.random.randint(N)
            centers[0] = data[first_idx]
            
            # 2. Select the remaining K-1 centers
            for k in range(1, self.K):
                # Compute distances from all points to the centers already selected
                distances = self.compute_distance(data, centers[:k])
                # Get the minimum distance squared for each point
                min_distances_sq = np.min(distances, axis=1) ** 2
                
                # Proportional probability distribution
                sum_dist = np.sum(min_distances_sq)
                if sum_dist == 0:
                    probabilities = np.ones(N) / N
                else:
                    probabilities = min_distances_sq / sum_dist
                
                next_idx = np.random.choice(N, p=probabilities)
                centers[k] = data[next_idx]
            return centers
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

    def compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.
        Fully vectorized using matrix math to avoid python loops.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features.
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        # Ensure centers is 2D
        centers = np.atleast_2d(centers)
        
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
        data_sq = np.sum(data ** 2, axis=1, keepdims=True)  # Shape (N, 1)
        centers_sq = np.sum(centers ** 2, axis=1)          # Shape (K,)
        dot_prod = np.dot(data, centers.T)                 # Shape (N, K)
        
        distances_sq = data_sq + centers_sq - 2 * dot_prod
        # Clip to 0 to prevent negative values due to floating-point precision issues
        distances = np.sqrt(np.maximum(distances_sq, 0))        return distances


    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """

        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments


    def compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments:
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """

        centers = np.zeros((self.K, data.shape[1]))
        for k in range(self.K):
            if np.sum(cluster_assignments == k) == 0:
                centers[k] = data[np.random.randint(data.shape[0])]
            else:
                centers[k] = np.mean(data[cluster_assignments == k], axis=0)
        return centers

    def compute_inertia(self, data, centers, cluster_assignments):
        """
        Compute the inertia (sum of squared distances to closest center).

        Arguments:
            data: array of shape (N, D)
            centers: array of shape (K, D)
            cluster_assignments: array of shape (N,)
        Returns:
            inertia (float): sum of squared distances to closest center
        """
        distances = self.compute_distance(data, centers)
        closest_distances = distances[np.arange(data.shape[0]), cluster_assignments]
        return np.sum(closest_distances ** 2)

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        Runs multiple restarts and selects the one with minimum inertia.

        Arguments:
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
        best_inertia = float('inf')
        best_centers = None
        best_assignments = None

        for attempt in range(self.n_init):
            centers = self.init_centers(data)
            cluster_assignments = None
            for i in range(max_iter):
                old_centers = centers.copy()
                distances = self.compute_distance(data, centers)
                cluster_assignments = self.find_closest_cluster(distances)
                centers = self.compute_centers(data, cluster_assignments)
                if np.allclose(old_centers, centers, atol=self.tolerance):
                    break
            
            if cluster_assignments is None:
                distances = self.compute_distance(data, centers)
                cluster_assignments = self.find_closest_cluster(distances)
                
            inertia = self.compute_inertia(data, centers, cluster_assignments)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_assignments = cluster_assignments

        return best_centers, best_assignments

    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments:
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns:
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """

        cluster_center_label = np.zeros(centers.shape[0])
        for i in range(len(centers)):
            assigned_labels = true_labels[cluster_assignments == i]
            if len(assigned_labels) == 0:
                cluster_center_label[i] = 0
            else:
                label = np.argmax(np.bincount(assigned_labels.astype(int)))
                cluster_center_label[i] = label
        return cluster_center_label

    def predict_with_centers(self, data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assign points in data to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns:
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """

        distances = self.compute_distance(data, centers)
        cluster_assignments = self.find_closest_cluster(distances)
        new_labels = cluster_center_label[cluster_assignments]
        return new_labels

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        centers, cluster_assignments = self.k_means(training_data, self.max_iters)
        self.centers = centers
        self.cluster_center_label = self.assign_labels_to_centers(centers, cluster_assignments, training_labels)
        pred_labels = self.predict_with_centers(training_data, self.centers, self.cluster_center_label)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = self.predict_with_centers(test_data, self.centers, self.cluster_center_label)
        return pred_labels