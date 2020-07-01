from typing import List

import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import scipy as sp

RANDOM_STATE = 42
FIG_SIZE = (10, 7)


def get_datas():
    return load_wine(return_X_y=True)
    # return train_test_split(features, target, test_size=0.30, random_state=RANDOM_STATE)


def mahalanobis(x, m, inv_cov=None):
    x_minus_m = x - m
    left_term = np.dot(x_minus_m, inv_cov)
    mahal = np.dot(left_term, x_minus_m.T)
    return mahal


def euclidean(x, m):
    x_minus_m = x - m
    euclid = np.dot(x_minus_m, x_minus_m.T)
    return euclid


def cluster_on_mahalanobis_distance(x, means: List[dict] = None, inv_covs=None):
    mahal = []
    for mean in means:
        mahal.append(mahalanobis(x, list(mean.values())[0], inv_covs[list(mean.keys())[0]]))
        # mahal.append(mahalanobis(x, list(mean.values())[0]))
    mi_dist = mahal.index(min(mahal))
    return list(means[mi_dist].keys())[0]


def cluster_on_euclidean_distance(x, means: List[dict] = None):
    euclid = []
    for mean in means:
        euclid.append(euclidean(x, list(mean.values())[0]))
        # mahal.append(mahalanobis(x, list(mean.values())[0]))
    min_dist = euclid.index(min(euclid))
    return list(means[min_dist].keys())[0]


def get_distance_euclidean(x=None, y=None, x_test=None):
    dist = []
    new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(x, y, test_size=0.7)
    if x_test is not None:
        new_X_test = x_test
    dict = get_label_dict(new_X_train, new_y_train)
    means = []
    for key in dict.keys():
        means.extend(get_means(dict[key], key))
    for i in range(0, len(new_X_test)):
        a = cluster_on_euclidean_distance(new_X_test[i], means=means)
        dist.append(a)
    get_confusion_matrix(y_test=new_y_test,dist=dist)
    return dist


def get_confusion_matrix(y_test, dist):
    print('Confusion Matrix :')
    print(confusion_matrix(y_test, dist))
    print('Accuracy Score :', accuracy_score(y_test, dist))


def get_covariance_inverse(dict_part, part):
    inv_cov = sp.linalg.inv(np.cov(dict_part.T))
    return {part: inv_cov}


def get_label_dict(X_train, y_train) -> dict:
    return {label: X_train[y_train == label] for label in np.unique(y_train)}


def get_mean(dict_part, part):
    m = np.mean(dict_part, axis=0)
    return {part: m}


def get_means(dict, part):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(dict)
    labels = kmeans.labels_
    new_dict = get_label_dict(dict, labels)
    means = []
    for key, value in new_dict.items():
        means.append(get_mean(value, part))
    return means


def get_distance(x=None, y=None, x_test=None):
    dist = []
    new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(x, y, test_size=0.7)
    if x_test is not None:
        new_X_test = x_test
    dict = get_label_dict(new_X_train, new_y_train)
    means = []
    inv_cov = {}
    for key in dict.keys():
        means.extend(get_means(dict[key], key))
        inv_cov={**inv_cov,**get_covariance_inverse(dict[key], key)}

    for i in range(0, len(new_X_test)):
        a = cluster_on_mahalanobis_distance(new_X_test[i], means=means, inv_covs=inv_cov)
        dist.append(a)
    get_confusion_matrix(y_test=new_y_test, dist=dist)
    return dist


x, y = get_datas()
# dist = get_distance(x, y)
get_distance_euclidean(x,y)

