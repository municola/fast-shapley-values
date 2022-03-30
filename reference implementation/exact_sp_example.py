import time
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from exact_sp import get_true_KNN, compute_single_unweighted_knn_class_shapley

# data = np.load('CIFAR10_resnet50-keras_features.npz')
# x_trn = np.vstack((data['features_training'], data['features_testing']))
# y_trn = np.hstack((data['labels_training'], data['labels_testing']))

# x_trn, y_trn = shuffle(x_trn, y_trn, random_state=0)

cifar = tf.keras.datasets.cifar10
(x_trn, y_trn), (x_tst, y_tst) = cifar.load_data()

# x_trn = np.reshape(x_trn, (-1, 2048))
x_trn = np.reshape(x_trn, (-1, 32*32*3))
x_tst, y_tst = x_trn[:100], y_trn[:100]
x_val, y_val = x_trn[100:210], y_trn[100:210]
x_trn, y_trn = x_trn[210:], y_trn[210:]

print("X-train.shape:", x_trn.shape)

# we are using 1-nn classifier
K = 1

start = time.time()
x_tst_knn_gt = get_true_KNN(x_trn, x_tst)
end1 = time.time() - start
print(end1)

start = time.time()
x_val_knn_gt = get_true_KNN(x_trn, x_val)
val_end1 = time.time() - start
print(val_end1)

start = time.time()
sp_gt = compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_tst_knn_gt, y_tst, K)
end2 = time.time() - start

start = time.time()
val_sp_gt = compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_val_knn_gt, y_val, K)
val_end2 = time.time() - start

print(end2)
print(val_end2)

print("time to get exact sp values for test set:", (end1 + end2) / len(x_tst))
print("time to get exact sp values for val set:", (val_end1 + val_end2) / len(x_val))

np.save('tst_exact_sp_gt', sp_gt)
np.save('val_exact_sp_gt', val_sp_gt)
