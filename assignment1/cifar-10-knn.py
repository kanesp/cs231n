#coding=utf-8
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
# print('Training data shape: ', X_train.shape)
# print('Training labels shape: ', y_train.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
X_train = np.reshape(X_train, (X_train.shape[0], -1))#X_train.shape:5000*32*32*3
X_test = np.reshape(X_test, (X_test.shape[0], -1))
# print(X_train.shape, X_test.shape,y_train.shape)
classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)
#
# dists = classifier.compute_distances_no_loops(X_test)
# y_test_pred = classifier.predict_labels(dists, k=1)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

#交叉验证
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

pass
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    k_to_accuracies[k] = np.zeros(num_folds)
    for i in range(num_folds):
        x_t = np.array(X_train_folds[:i] + X_train_folds[i + 1:])  # 剩下为训练集
        y_t = np.array(y_train_folds[:i] + y_train_folds[i + 1:])
        x_t = x_t.reshape(X_train_folds[i].shape[0] * 4, -1)
        y_t = y_t.reshape(y_train_folds[i].shape[0] * 4, -1)

        x_te = np.array(X_train_folds[i])  # 测试集
        y_te = np.array(y_train_folds[i])

        classifier.train(x_t, y_t)
        dists_ = classifier.compute_distances_no_loops(x_te)
        y_pred = classifier.predict_labels(dists_, k)
        # Compute and print the fraction of correctly predicted examples
        num_correct = np.sum(y_pred == y_te)
        accuracy = float(num_correct) / num_test
        k_to_accuracies[k][i] = accuracy
pass
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))