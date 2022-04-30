import time
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from exact_sp import get_true_KNN, compute_single_unweighted_knn_class_shapley

""" Uncomment to load data from web """
# cifar = tf.keras.datasets.cifar10
# (x_trn, y_trn), (x_tst, y_tst) = cifar.load_data()

# np.save("x_trn", x_trn)
# np.save("y_trn", y_trn)

# """ Uncomment to load data from local files """
# x_base = np.load("../data/features/cifar10/train_features.npy")
# y_trn = np.load("../data/features/cifar10/train_labels.npy")

# print("X-base.shape:", x_base.shape)

# # x_base = np.reshape(x_base, (-1, 2048))
# # x_base = np.reshape(x_base, (-1, 32*32*3))
# x_val, y_val = x_base[100:210], y_trn[100:210]
# x_trn, y_trn = x_base[:5], y_trn[:5]
# x_tst, y_tst = x_base[:10], y_trn[:10]
# # x_trn, y_trn = x_trn[210:], y_trn[210:]

# print("X-train:", x_trn)

# print("X-train.shape:", x_trn.shape)
# print("X-test.shape:", x_tst.shape)
# # print("Y-train.shape:", y_trn.shape)
# # we are using 1-nn classifier
# K = 1

# start = time.time()
# x_tst_knn_gt = get_true_KNN(x_trn, x_tst)
# print(x_tst_knn_gt.shape, x_tst_knn_gt)
# end1 = time.time() - start
# print(end1)


# start = time.time()
# sp_gt = compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_tst_knn_gt, y_tst, K)
# end2 = time.time() - start
# print(sp_gt)

# data = np.load('CIFAR10_resnet50-keras_features.npz')
# x_trn = np.vstack((data['features_training'], data['features_testing']))
# y_trn = np.hstack((data['labels_training'], data['labels_testing']))

x_trn = np.load("../data/features/cifar10/train_features.npy")
y_trn = np.load("../data/features/cifar10/train_labels.npy")

print(y_trn[:10])

# x_trn, y_trn = shuffle(x_trn, y_trn, random_state=0)

# x_trn = x_trn[:200]

x_trn = np.reshape(x_trn, (-1, 2048))
# x_tst, y_tst = x_trn[:50], y_trn[:50]
x_tst, y_tst = x_trn[:5], y_trn[:5]
x_val, y_val = x_trn[100:1100], y_trn[100:1100]
# x_trn, y_trn = x_trn[49000:], y_trn[49000:]
x_trn, y_trn = x_trn[49990:], y_trn[49990:]

# we are using 1-nn classifier
K = 1

print("x_trn:")
print(x_trn)

print()
print("x_tst:")
print(x_tst)

print()
print("y_trn:")
print(y_trn)



print()
print("y_tst:")
print(y_tst)

start = time.time()
x_tst_knn_gt = get_true_KNN(x_trn, x_tst)
end1 = time.time() - start
# print(end1)

print("x_tst_knn_gt:")
print(x_tst_knn_gt)

# start = time.time()
# x_val_knn_gt = get_true_KNN(x_trn, x_val)
# val_end1 = time.time() - start
# print(val_end1)

start = time.time()
sp_gt = compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_tst_knn_gt, y_tst, K)
end2 = time.time() - start

# start = time.time()
# val_sp_gt = compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_val_knn_gt, y_val, K)
# val_end2 = time.time() - start

# print(end2)
# print(val_end2)

print()
print("Shapley Values:")
print(sp_gt)
# print("time to get exact sp values for test set:", (end1 + end2) / len(x_tst))
# print("time to get exact sp values for val set:", (val_end1 + val_end2) / len(x_val))

# np.save('tst_exact_sp_gt', sp_gt)
# np.save('val_exact_sp_gt', val_sp_gt)