import json

import matplotlib.pyplot as plt
import torch


def changeType(x, target='Tensor'):
    if type(x).__name__ != target:
        if target == 'Tensor':
            return torch.tensor(x)

def plot_history(history, file=None, fig=None, detailed=True, label=None):
    """Plot the training history of the model.
    """

    epochs = history['epochs']
    losses = history['losses']
    errors = history['errors']

    if not fig:
        plt.rcParams['figure.figsize'] = [15, 10] if errors else [15, 5]
        fig, _ = plt.subplots(2, 1) if errors else plt.subplots(1, 1)
        fig.tight_layout(pad=4.0)
        fig.suptitle(f'Training loss and H1-error')
    axs = fig.axes

    # Plot the loss
    axs[0].plot(epochs, losses, label=label if label else 'Total Loss')
    axs[0].set(xscale='linear', yscale='log', xlabel='Epoch', ylabel='Loss')

    # Plot the error(s)
    if len(axs) == 2 and errors:
        axs[1].plot(epochs, errors['tot'],
                    label= label if label else 'Solution H1-error')
        if detailed:
            # axs[1].plot(epochs, errors['sol']
            #       , label=label+': Solution L2-error' if label else 'Solution L2-error')
            axs[1].plot(epochs, errors['der'],
                    label=label if label else 'Derivative L2-error')
        axs[1].set(xscale='linear', yscale='log', xlabel='Epoch', ylabel='Error')

    for ax in axs:
        ax.grid(which='both')
        ax.legend()

    if file: plt.savefig(file)

def plot_validation(xpts, upts, rpts, title='Validation', subscript='', file=None):

    upts_re = upts[0]
    upts_im = upts[1]
    rpts_re = rpts[0]
    rpts_im = rpts[1]

    plt.rcParams['figure.figsize'] = [15, 7]
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=4.0)
    fig.suptitle(title)

    axs[0, 0].plot(xpts, upts_re, label='$u'+subscript+'(x)$')
    axs[0, 0].plot(xpts, rpts_re, label='$u^N'+subscript+'(x)$')
    axs[0, 0].set(xlabel='x', ylabel='$Re[u'+subscript+'(x)]$')
    axs[0, 1].plot(xpts, upts_im, label='$u'+subscript+'(x)$')
    axs[0, 1].plot(xpts, rpts_im, label='$u^N'+subscript+'(x)$')
    axs[0, 1].set(xlabel='x', ylabel='$Im[u'+subscript+'(x)]$')

    # axs[1, 0].errorbar(xpts, upts_re, yerr=(upts_re - rpts_re), ecolor='black', label='$u'+subscript+'(x)$')
    # axs[1, 0].set(xlabel='x', ylabel='$Re[u'+subscript+'(x)]$')
    # axs[1, 1].errorbar(xpts, upts_im, yerr=(upts_im - rpts_im), ecolor='black', label='$u'+subscript+'(x)$')
    # axs[1, 1].set(xlabel='x', ylabel='$Im[u'+subscript+'(x)]$')

    axs[1, 0].plot(xpts, upts_re - rpts_re)
    axs[1, 0].set(xlabel='x', ylabel='$Re[u'+subscript+'(x)-u^N'+subscript+'(x)]$')
    axs[1, 1].plot(xpts, upts_im - rpts_im)
    axs[1, 1].set(xlabel='x', ylabel='$Im[u'+subscript+'(x)-u^N'+subscript+'(x)]$')

    for row in axs:
        for ax in row:
            ax.grid()
            if len(ax.get_legend_handles_labels()[1]) > 0:
                ax.legend()

    if file: plt.savefig(file)

def plot_histories(dirs, file=None, fig=None, plot_error=True):
    """Plot the training histories of the models in the given directories.
    """

    if not fig:
        plt.rcParams['figure.figsize'] = [15, 10] if plot_error else [15, 5]
        fig, _ = plt.subplots(2, 1) if plot_error else plt.subplots(1, 1)
        fig.tight_layout(pad=4.0)
        fig.suptitle(f'Training loss and H1-error')

    for dir in dirs:
        with open(dir+'-info.json', 'r') as f:
            info = json.load(f)
        with open(dir+'-train_history.json', 'r') as f:
            history = json.load(f)

        plot_history(
            history, file=None, fig=fig, detailed=False,
            label='D='+ str(info['model']['depth']) + ', N='+ str(info['model']['width']) + ', K=' + str(info['model']['testfuncs'])
            # label='$k=%.1f$' % info['equation']['k'],
            )

    for ax in fig.axes:
        ax.grid()
        ax.legend()

    if file: plt.savefig(file)
