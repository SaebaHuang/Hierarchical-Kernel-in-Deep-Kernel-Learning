# encoding=UTF-8
from dataloader import get_dataloader
from Libs.thundersvm_py import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
import sys
import os
from time import strftime
from utils import logger

if __name__ == '__main__':

    # Process the args
    sys_args = ['python']
    sys_args.extend(sys.argv)

    dataset_dir_pattern = './dataset/{}/'
    dataset_name = 'cifar10'
    dataset_dir = dataset_dir_pattern.format(dataset_name)

    # load data
    dataloader = get_dataloader(dataset_name)
    X_train, Y_train, X_test, Y_test = dataloader(dataset_dir)
    X = np.concatenate([X_train, X_test], axis=0)
    Y = np.concatenate([Y_train, Y_test], axis=0)

    # Perform required task
    kernel_list = ['rbf', 'rbf_1', 'rbf_2', 'rbf_3']
    param_search_grid = {
        'gamma': [np.power(2.0, log_gamma) for log_gamma in np.arange(-19, -7, 1)],
    }

    # cur timestamp
    timestamp = strftime('%Y_%m_%d_%H_%M_%S')

    # root
    dir_root = f'./results/cifar10/{dataset_name}_{timestamp}/'
    if not os.path.exists(dir_root):
        os.makedirs(dir_root)

    # logger pattern
    logger_dir_pattern = dir_root + '{}_{}_GS_{}.txt'

    # figure pattern
    figure_dir_pattern = dir_root + '{}_{}_GS_{}.pdf'

    for kernel in kernel_list:
        # Create logger file and record the time of starting
        timestamp = strftime('%Y_%m_%d_%H_%M_%S')
        logger_dir = logger_dir_pattern.format(dataset_name, kernel, timestamp)
        logger('Start: {}'.format(timestamp), logger_dir)

        # Perform GirdSearch for the best hyperparameters
        svc = SVC(kernel=kernel, C=1.0, max_iter=100)
        clf = GridSearchCV(estimator=svc, param_grid=param_search_grid, verbose=3, return_train_score=True,
                           scoring='accuracy', n_jobs=12, cv=[(list(range(0, 50000)), list(range(50000, 60000)))])
        clf.fit(X=X, y=Y)

        time_train = np.sum(clf.cv_results_['mean_fit_time'])
        acc_train = clf.cv_results_['mean_train_score']
        acc_test = clf.cv_results_['mean_test_score']

        logger('Best params: {}'.format(clf.best_params_), logger_dir)
        logger('Best score: {}'.format(clf.best_score_), logger_dir)
        logger('Train time: {}'.format(time_train), logger_dir)
        logger('Train acc: {}'.format(acc_train), logger_dir)
        logger('Train acc max: {}, mean: {}, std: {}'.format(acc_train.max(), acc_train.mean(), acc_train.std()),
               logger_dir)
        logger('Test acc: {}'.format(acc_test), logger_dir)
        logger('Test acc max:{}, mean: {}, std: {}'.format(acc_test.max(), acc_test.mean(), acc_test.std()), logger_dir)
        logger('CV Results:\n{}'.format(clf.cv_results_), logger_dir)

        test_score = clf.cv_results_['mean_test_score'] * 100
        st = ''
        for m in test_score:
            st = st + '{:.2f}\\% & '.format(m)
        logger(st, logger_dir)

        # Record the time of ending
        end_timestamp = strftime('%Y_%m_%d_%H_%M_%S')
        logger('End: {}'.format(end_timestamp), logger_dir)
