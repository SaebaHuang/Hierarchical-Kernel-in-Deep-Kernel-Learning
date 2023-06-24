# encoding=UTF-8
from dataloader import get_dataloader
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn import svm
import sys
import os
from time import strftime
from utils import logger

dataset_list = ['bodyfat', 'pyrim', 'triazines']


def hierarchical_exp_kernel(given_data, input_data, hyper_lambda, n_layers=0):
    diff = given_data[:, np.newaxis] - input_data
    diff_abs = np.abs(diff)
    norm_1 = np.sum(diff_abs, axis=2)
    kernel_mat = np.exp(-hyper_lambda * norm_1)

    e_n = 0.0

    for layer_id in range(0, n_layers):
        e_n = np.exp(e_n)
        kernel_mat = np.exp(e_n * (kernel_mat - 1.0))

    return kernel_mat


if __name__ == '__main__':

    # Process the args
    sys_args = ['python']
    sys_args.extend(sys.argv)

    dataset_dir_pattern = './dataset/libsvm_regression_tasks/{}.txt'

    dataset_id = int(sys_args[2])
    assert dataset_id < 3

    dataset_name = dataset_list[dataset_id]
    dataset_dir = dataset_dir_pattern.format(dataset_name)

    assert len(sys_args) == 4
    fold_num = int(sys_args[3])  # fold number for CV

    # load data
    dataloader = get_dataloader(dataset_name)
    X, Y = dataloader(dataset_dir)

    # Perform required task
    kernel_list = ['exp', 'exp_1', 'exp_2', 'exp_3']

    # cur timestamp
    timestamp = strftime('%Y_%m_%d_%H_%M_%S')

    # root
    dir_root = f'./results/LIBSVM/{dataset_name}_{timestamp}/'
    if not os.path.exists(dir_root):
        os.makedirs(dir_root)

    # logger pattern
    logger_dir_pattern = dir_root + '{}_{}_GSCV_F_{}_{}.txt'

    for kernel in kernel_list:
        # Create logger file and record the time of starting
        timestamp = strftime('%Y_%m_%d_%H_%M_%S')
        logger_dir = logger_dir_pattern.format(dataset_name, kernel, fold_num, timestamp)
        logger('Start: {}'.format(timestamp), logger_dir)

        # Perform GirdSearch for the best hyperparameters
        kernel_id = kernel_list.index(kernel)
        lambda_list = [np.power(2.0, log_lambda) for log_lambda in np.arange(-13, -2, 1)]

        train_mean = []
        test_mean = []
        train_std = []
        test_std = []
        for hyper_l in lambda_list:
            svr = svm.SVR(
                C=1.0,
                epsilon=0.001,
                kernel=lambda x_i, x_j: hierarchical_exp_kernel(x_i, x_j, hyper_l, kernel_id),
                max_iter=100
            )
            kf = KFold(n_splits=fold_num, shuffle=True, random_state=321)
            cv_results = cross_validate(svr, X, Y, cv=list(kf.split(Y)), scoring='neg_root_mean_squared_error',
                                        n_jobs=5, verbose=True,
                                        return_train_score=True)
            train_score = -cv_results['train_score']
            test_score = -cv_results['test_score']
            train_mean.append(train_score.mean())
            train_std.append(train_score.std())
            test_mean.append(test_score.mean())
            test_std.append(test_score.std())

        train_mean = np.array(train_mean)
        test_mean = np.array(test_mean)
        train_std = np.array(train_std)
        test_std = np.array(test_std)


        logger('Best params: {}'.format(lambda_list[test_mean.argmin()]), logger_dir)
        logger('Best score: {}'.format(test_mean.min()), logger_dir)

        st = ''
        for m, s in zip(test_mean, test_std):
            st = st + '{:.4f}$\\pm${:.4f} & '.format(m, s)
        logger(st, logger_dir)

        # Record the time of ending
        end_timestamp = strftime('%Y_%m_%d_%H_%M_%S')
        logger('End: {}'.format(end_timestamp), logger_dir)
