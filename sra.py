import datasets_list
import numpy as np
from experiments_utils import saveScores

def SRA(W, xi=0.05):
    n_rows = len(W)
    d = np.sum(W, axis=1)
    d_sqrt = np.sqrt(d)

    laplacian = np.identity(n_rows) - (1 / d_sqrt) * np.matmul(W, np.diag(1 / d_sqrt))

    np_eigenvalues, np_eigenvectors = np.linalg.eig(laplacian)
    np_eigen_sorted = sorted(zip(np_eigenvalues, np_eigenvectors.T), key=lambda x: x[0].real)
    first_np_eigenvector = np_eigen_sorted[1][1]

    z = d_sqrt * first_np_eigenvector

    Cp = [i for i in z if i >= 0]
    Cn = [i for i in z if i < 0]

    if min(len(Cp) / n_rows, len(Cn) / n_rows) >= xi:
        mFLAG = 1
        f = max(np.abs(z)) - np.abs(z)
    elif len(Cp) > len(Cn):
        mFLAG = 0
        f = -z
    else:
        mFLAG = 0
        f = z

    return f, mFLAG

for [benchmark_path, benchmark_name, y_benchmark_name, benchmark_data_lines, benchmark_data_columns, benchmark_object_byte_size] in datasets_list.datasets_realistic:
    f=open(y_benchmark_name, "r")
    lines=f.readlines()
    test_y_outlier=[]
    test_y_not_outlier=[]
    test_y_local_outlier=[]
    test_y_global_outlier=[]
    test_y_collective_outlier=[]
    for x in lines:
        x_spl = x.split(';')[1]
        if x_spl.rstrip() == '0':
            test_y_not_outlier.append(int(x.split(';')[0]))
        if x_spl.rstrip() == '1':
            test_y_outlier.append(int(x.split(';')[0]))
            test_y_local_outlier.append(int(x.split(';')[0]))
        if x_spl.rstrip() == '2':
            test_y_outlier.append(int(x.split(';')[0]))
            test_y_global_outlier.append(int(x.split(';')[0]))
        if x_spl.rstrip() == '3':
            test_y_outlier.append(int(x.split(';')[0]))
            test_y_collective_outlier.append(int(x.split(';')[0]))
    f.close()

    aucpr_list = []
    aucroc_list = []
    overall_aucpr_list = []
    overall_aucroc_list = []
    local_aucpr_list = []
    local_aucroc_list = []
    global_aucpr_list = []
    global_aucroc_list = []
    collective_aucpr_list = []
    collective_aucroc_list = []

    data = np.loadtxt("km_" + benchmark_name + ".txt", dtype=float, delimiter=';')
    data = (data - np.min(data)) / np.ptp(data)

    ids = np.loadtxt(y_benchmark_name, dtype=int, delimiter=';')
    ids = ids[:, 0]

    rank, flag = SRA(data)
    scores = np.column_stack((ids, rank))
    scores = scores[scores[:, 1].argsort()]

    saveScores(scores, scores.shape[0], test_y_outlier, test_y_not_outlier, overall_aucpr_list, overall_aucroc_list)
    if flag == 1:
        saveScores(scores, scores.shape[0], test_y_local_outlier, test_y_not_outlier + test_y_global_outlier + test_y_collective_outlier,
                    local_aucpr_list, local_aucroc_list)
        saveScores(scores, scores.shape[0], test_y_global_outlier, test_y_not_outlier + test_y_local_outlier + test_y_collective_outlier,
                    global_aucpr_list, global_aucroc_list)
    if flag == 0:
        saveScores(scores, scores.shape[0], test_y_collective_outlier, test_y_not_outlier + test_y_local_outlier + test_y_global_outlier,
                    collective_aucpr_list, collective_aucroc_list)

    aucpr_list.append(overall_aucpr_list)
    aucroc_list.append(overall_aucroc_list)
    aucpr_list.append(local_aucpr_list)
    aucroc_list.append(local_aucroc_list)
    aucpr_list.append(global_aucpr_list)
    aucroc_list.append(global_aucroc_list)
    aucpr_list.append(collective_aucpr_list)
    aucroc_list.append(collective_aucroc_list)
    aucpr_list.append(flag)
    aucroc_list.append(flag)

    np.savetxt(("./stats/" + benchmark_name + "_pxr.csv"), aucpr_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "_roc.csv"), aucroc_list, fmt='%s', delimiter=',')
