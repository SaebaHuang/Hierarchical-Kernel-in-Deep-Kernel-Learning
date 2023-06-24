# encoding=UTF-8
from dataloader import get_dataloader
from Libs.thundersvm_py import SVR
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
import sys
import os
from time import strftime
from utils import logger

dataset_list = ['elevators', 'kin40k', 'servo']

if __name__ == '__main__':

    # Process the args
    sys_args = ['python']
    sys_args.extend(sys.argv)

    dataset_dir_pattern = '{}'

    dataset_id = int(sys_args[2])
    assert dataset_id < 3

    dataset_name = dataset_list[dataset_id]
    dataset_dir = dataset_dir_pattern.format(dataset_name)

    assert len(sys_args) >= 4
    fold_num = int(sys_args[3])  # fold number for CV

    # load data
    dataloader = get_dataloader(dataset_name)
    X, Y = dataloader(dataset_dir)

    # Perform required task
    kernel_list = ['rbf', 'rbf_1', 'rbf_2', 'rbf_3']
    param_search_grid = {
        'C': [1.0],
        'gamma': [np.power(2.0, log_gamma) for log_gamma in np.arange(-8, 3, 1)],
        'epsilon': [0.001]
    }

    # cur timestamp
    timestamp = strftime('%Y_%m_%d_%H_%M_%S')

    # root
    dir_root = f'./results/UCI/{dataset_name}_{timestamp}/'
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
        svr = SVR(kernel=kernel, max_iter=100)
        kf = KFold(n_splits=fold_num, shuffle=True, random_state=321)
        clf = GridSearchCV(estimator=svr, param_grid=param_search_grid, verbose=3,
                           scoring='neg_root_mean_squared_error', n_jobs=8, cv=list(kf.split(Y)))
        clf.fit(X=X, y=Y)
        logger('Best params: {}'.format(clf.best_params_), logger_dir)
        logger('Best score: {}'.format(clf.best_score_), logger_dir)
        logger('CV Results:\n{}'.format(clf.cv_results_), logger_dir)

        test_score_mean = -clf.cv_results_['mean_test_score']
        test_score_std = clf.cv_results_['std_test_score']
        st = ''
        for m, s in zip(test_score_mean, test_score_std):
            st = st + '{:.4f}$\\pm${:.4f} & '.format(m, s)
        logger(st, logger_dir)

        # Record the time of ending
        end_timestamp = strftime('%Y_%m_%d_%H_%M_%S')
        logger('End: {}'.format(end_timestamp), logger_dir)
