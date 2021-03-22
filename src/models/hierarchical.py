import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from src.models.recommender import Recommender

class HierarchicalClustering(Recommender):
    def __init__(
        self,
        n_clusters: int = None,
        distance_threshold: float = 0.0
    ):
        super(HierarchicalClustering, self).__init__()
        if n_clusters is None:
            self.model = AgglomerativeClustering(
                distance_threshold=distance_threshold,
                n_clusters=None
                )
        else:
            self.model = AgglomerativeClustering(
                distance_threshold=None,
                n_clusters=n_clusters
                )


    def fit(
        self,
        interactions: np.ndarray
    ):
        self.model.fit(interactions)


    def plot_dendrogram(self, truncate_mode='level', p=3):
        # create the counts of samples under each node
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.model.children_, self.model.distances_,
                                        counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix)

        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()
