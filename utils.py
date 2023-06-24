# encoding=UTF-8
import matplotlib.pyplot as plt
import sys
from os.path import exists
from sklearn.inspection import DecisionBoundaryDisplay


def logger(msg, txt_dir):
    default_std_out = sys.stdout
    if not exists(txt_dir):
        experiment_results_logger = open(txt_dir, 'x')
    else:
        experiment_results_logger = open(txt_dir, 'a')
    sys.stdout = experiment_results_logger
    print(msg)
    sys.stdout = default_std_out
    experiment_results_logger.close()
    print(msg)
    return


def plot_decision_boundary(clf, x, y, kernel, fig_dir):
    # plot contour
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        x,
        response_method='predict',
        cmap=plt.cm.coolwarm,
        ax=ax,
        xlabel='x',
        ylabel='y',
        alpha=0.8,
    )
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")


    # save figure
    title = kernel.replace('rbf', 'G')
    if title == 'G':
        title = 'G_0'
    ax.set_title(title)
    plt.minorticks_off()
    plt.savefig(fig_dir, format='pdf', dpi=400)

    return
