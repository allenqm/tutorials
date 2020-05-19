# implement kmeans in pytorch
# solve a linear system of equations in pytoch
# other matrix operations?
import torch


class KmeansTorch():
    def __init__(self):
        pass

    def fit(self, X, n_clusters):
        centroids = torch.rand(size=(n_clusters, X.shape[1]))
        self.centroids = centroids


#training loop
n_clusters=5
X = torch.rand(size=(100,5))
centroids = torch.rand(size=(n_clusters, 5))
tolerance = 1e-4
iter_limit = 0

while True:
    # find closest centroid for each 
    sample_by_cluster_distance_matrix = ((X - centroids.unsqueeze(1))**2).sum(-1).T

    choice_cluster = torch.argmin(sample_by_cluster_distance_matrix, dim=1)

    #update centroids
    centroids_pre = centroids.clone()

    for index in range(5): #n_clusters
        selected = torch.nonzero(choice_cluster == index).squeeze()

        selected = torch.index_select(X, 0, selected)

        centroids[index] = selected.mean(dim=0)

    # decide to iterate
    center_shift = torch.sum(
        torch.sqrt(
            torch.sum((centroids - centroids_pre) ** 2, dim=1)
        ))

    if center_shift < tolerance:
        break

    if iter_limit !=0 and iteration >= iter_limit:
        break



n_clusters=5
X = torch.rand(size=(100,5))
centroids = torch.rand(size=(n_clusters, 5))
tolerance = 1e-4
iter_limit = 0
iteration = 0
while True:
    iteration = iteration + 1
    initialize_centroids(n_clusters=n_clusters)
    find_closest_centroid(X, centroids)
    update_centroids(X, centroids, n_clusters)
    centroids_pre = centroids.clone()
    if decide_whether_to_iterate(centroids, centroids_pre, iteration, tolerance, iter_limit) == 1:
        break


def initialize_centroids(n_clusters):
    centroids = torch.rand(size=(n_clusters, 5))
    return centroids  


def find_closest_centroid(X, centroids):
    sample_by_cluster_distance_matrix = ((X - centroids.unsqueeze(1))**2).sum(-1).T

    choice_cluster = torch.argmin(sample_by_cluster_distance_matrix, dim=1)    

    return choice_cluster

def update_centroids(X, centroids, n_clusters):
    for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()

            selected = torch.index_select(X, 0, selected)

            centroids[index] = selected.mean(dim=0)    

    return centroids

def decide_whether_to_iterate(centroids, centroids_pre, iteration, tolerance, iter_limit):
    center_shift = torch.sum(
        torch.sqrt(
            torch.sum((centroids - centroids_pre) ** 2, dim=1)
        ))

    if center_shift < tolerance:
        return 1

    if iter_limit !=0 and iteration >= iter_limit:
        return 1
