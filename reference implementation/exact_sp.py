import numpy as np
from tqdm import tqdm


def get_true_KNN(x_trn, x_tst):
    N = x_trn.shape[0]
    N_tst = x_tst.shape[0]
    print(N, N_tst)
    x_tst_knn_gt = np.zeros((N_tst, N))

    for i_tst in tqdm(range(N_tst)):
        dist_gt = np.zeros(N)
        for i_trn in range(N):
            dist_gt[i_trn] = np.linalg.norm(x_trn[i_trn, :] - x_tst[i_tst, :], 2)
            # print("x_trn.shape: ", x_trn[i_trn,:].shape, "x_tst.shape:", x_tst.shape)
        # print("Dist_gt:")
        # print(dist_gt)
        x_tst_knn_gt[i_tst, :] = np.argsort(dist_gt)
    return x_tst_knn_gt.astype(int)


def compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_tst_knn_gt, y_tst, K):
    N = x_trn.shape[0]
    N_tst = x_tst_knn_gt.shape[0]
    sp_gt = np.zeros((N_tst, N))
    for j in tqdm(range(N_tst)):
        # print("Iteration: j=", j)
        sp_gt[j, x_tst_knn_gt[j, -1]] = (y_trn[x_tst_knn_gt[j, -1]] == y_tst[j]) / N
        for i in np.arange(N - 2, -1, -1):
            # print("  Iteration: i=", i)
            # print("    s_j..:", sp_gt[j, x_tst_knn_gt[j, i + 1]])
            # print("    diff:", 
            # (int(y_trn[x_tst_knn_gt[j, i]] == y_tst[j]) - int(y_trn[x_tst_knn_gt[j, i + 1]] == y_tst[j])),
            # "(",
            # y_trn[x_tst_knn_gt[j, i]],
            # y_tst[j],
            # "), (",
            # y_trn[x_tst_knn_gt[j, i + 1]],
            # y_tst[j], 
            # ")"
            # )
            # print("    min_:", min([K, i + 1]))

            sp_gt[j, x_tst_knn_gt[j, i]] = sp_gt[j, x_tst_knn_gt[j, i + 1]] + \
                                           (int(y_trn[x_tst_knn_gt[j, i]] == y_tst[j]) -
                                            int(y_trn[x_tst_knn_gt[j, i + 1]] == y_tst[j])) / K * min([K, i + 1]) / (
                                                       i + 1)
    return sp_gt
