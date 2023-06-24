# encoding=UTF-8
import os
from dataloader import get_dataloader
from Libs.thundersvm_py import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
import sys
from time import strftime
from utils import logger, plot_decision_boundary


if __name__ == '__main__':

    # Process the args
    sys_args = ['python']
    sys_args.extend(sys.argv)

    dataset_dir = None
    dataset_name = 'moons'

    task_type = sys_args[2]  # GridSearchCV or Evaluate
    assert task_type in ['GridSearch', 'Evaluate']

    # load data
    dataloader = get_dataloader(dataset_name)
    X_train, X_test, Y_train, Y_test = dataloader(dataset_dir, n_samples=1000)
    X = np.concatenate([X_train, X_test], axis=0)
    Y = np.concatenate([Y_train, Y_test], axis=0)

    # Perform required task
    if task_type == 'GridSearch':
        kernel_list = ['rbf', 'rbf_1', 'rbf_2', 'rbf_3']
        param_search_grid = {
            'C': [1],
            'gamma': [np.power(2.0, log_gamma) for log_gamma in np.arange(-5, 11, 1)],
        }

        # cur timestamp
        timestamp = strftime('%Y_%m_%d_%H_%M_%S')

        # root
        dir_root = f'./results/moons/{dataset_name}_{timestamp}/'
        if not os.path.exists(dir_root):
            os.makedirs(dir_root)

        # logger pattern
        logger_dir_pattern = dir_root + '{}_{}_GS_{}.txt'

        for kernel in kernel_list:
            # Create logger file and record the time of starting
            timestamp = strftime('%Y_%m_%d_%H_%M_%S')
            logger_dir = logger_dir_pattern.format(dataset_name, kernel, timestamp)
            logger('Start: {}'.format(timestamp), logger_dir)

            # Perform GirdSearch for the best hyperparameters
            svr = SVC(kernel=kernel, max_iter=100)
            clf = GridSearchCV(estimator=svr, param_grid=param_search_grid, verbose=3,
                               scoring='accuracy',
                               n_jobs=5, return_train_score=True,
                               cv=[(list(range(0, X_train.shape[0])), list(range(X_train.shape[0], X_train.shape[0]+X_test.shape[0])))]
                               )
            clf.fit(X=X, y=Y)

            time_train = np.sum(clf.cv_results_['mean_fit_time'])

            logger('Best params: {}'.format(clf.best_params_), logger_dir)
            logger('Best score: {}'.format(clf.best_score_), logger_dir)
            logger('Train time: {}'.format(time_train), logger_dir)
            logger('CV Results:\n{}'.format(clf.cv_results_), logger_dir)

            test_score = clf.cv_results_['mean_test_score'] * 100
            train_score = clf.cv_results_['mean_train_score'] * 100
            all_score = train_score * X_train.shape[0] / X.shape[0] + test_score * X_test.shape[0] / X.shape[0]
            st = ''
            for m in all_score:
                st = st + '{:.1f}\\% & '.format(m)
            logger(st, logger_dir)

            # Record the time of ending
            end_timestamp = strftime('%Y_%m_%d_%H_%M_%S')
            logger('End: {}'.format(end_timestamp), logger_dir)
    else:
        assert len(sys_args) == 6  # kernel, log C and log gamma
        kernel = sys_args[3]
        log_C = sys_args[4]
        log_gamma = sys_args[5]

        C_float = np.power(2.0, float(log_C))
        gamma_float = np.power(2.0, float(log_gamma))

        # cur timestamp
        timestamp = strftime('%Y_%m_%d_%H_%M_%S')

        # root
        dir_root = f'./results/moons/{dataset_name}_{timestamp}/'
        if not os.path.exists(dir_root):
            os.makedirs(dir_root)

        # logger pattern
        logger_dir_pattern = dir_root + '{}_{}_GS_{}.txt'

        # figure pattern
        figure_dir_pattern = dir_root + '{}_{}_DecisionBoundary_{}.pdf'

        # Create logger file and record the time of starting
        timestamp = strftime('%Y_%m_%d_%H_%M_%S')
        logger_dir = logger_dir_pattern.format(dataset_name, kernel, timestamp)
        logger('Start: {}'.format(timestamp), logger_dir)
        logger('C: {}, gamma: {}'.format(C_float, gamma_float), logger_dir)

        svc = SVC(
            kernel=kernel,
            C=C_float,
            gamma=gamma_float,
            max_iter=100, verbose=True
        )
        svc.fit(X_train, Y_train)
        predictions = svc.predict(X)
        acc = np.sum(predictions == Y) / Y.shape[0]
        logger('Results:\n{}'.format(acc), logger_dir)

        # Plot decision boundary
        figure_dir = figure_dir_pattern.format(dataset_name, kernel, timestamp)
        plot_decision_boundary(svc, X, Y, kernel, figure_dir)

        # Record the time of ending
        end_timestamp = strftime('%Y_%m_%d_%H_%M_%S')
        logger('End: {}'.format(end_timestamp), logger_dir)
