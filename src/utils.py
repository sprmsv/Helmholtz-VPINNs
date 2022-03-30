import matplotlib.pyplot as plt
import torch


def changeType(x, target='Tensor'):
    if type(x).__name__ != target:
        if target == 'Tensor':
            return torch.tensor(x)

def plot_history(history, file=None):

    epochs = history['epochs']
    losses = history['losses']
    errors = history['errors']

    plt.rcParams['figure.figsize'] = [15, 10] if errors else [15, 5]
    fig, axs = plt.subplots(2, 1) if errors else plt.subplots(1, 1)
    fig.tight_layout(pad=4.0)
    fig.suptitle(f'Training loss and H1-error')

    if errors:
        axs[0].plot(epochs, losses, label='Total Loss')
        axs[1].plot(epochs, errors['tot'], label='Solution H1-error')
        # axs[1].plot(epochs, errors['sol'], label='Solution L2-error')
        axs[1].plot(epochs, errors['der'], label='Derivative L2-error')
        axs[0].set(xscale='linear', yscale='log', xlabel='Epoch', ylabel='Loss')
        axs[1].set(xscale='linear', yscale='log', xlabel='Epoch', ylabel='Error')
        for ax in axs:
            ax.grid(which='both')
            ax.legend()
    else:
        axs.plot(epochs, losses, label='Total Loss')
        axs.set(xscale='linear', yscale='log', xlabel='Epoch', ylabel='Loss')
        axs.grid(which='both')
        axs.legend()

    if file: plt.savefig(file)

def plot_validation(xpts, upts, rpts, title='Validation', subscript='', file=None):

    upts_re = upts[0]
    upts_im = upts[1]
    rpts_re = rpts[0]
    rpts_im = rpts[1]

    plt.rcParams['figure.figsize'] = [15, 10]
    fig, axs = plt.subplots(3, 2)
    fig.tight_layout(pad=4.0)
    fig.suptitle(title)

    axs[0, 0].plot(xpts, upts_re, label='$u'+subscript+'(x)$')
    axs[0, 0].plot(xpts, rpts_re, label='$u^N'+subscript+'(x)$')
    axs[0, 0].set(xlabel='x', ylabel='$Re[u'+subscript+'(x)]$')
    axs[0, 1].plot(xpts, upts_im, label='$u'+subscript+'(x)$')
    axs[0, 1].plot(xpts, rpts_im, label='$u^N'+subscript+'(x)$')
    axs[0, 1].set(xlabel='x', ylabel='$Im[u'+subscript+'(x)]$')

    axs[1, 0].errorbar(xpts, upts_re, yerr=(upts_re - rpts_re), ecolor='black', label='$u'+subscript+'(x)$')
    axs[1, 0].set(xlabel='x', ylabel='$Re[u'+subscript+'(x)]$')
    axs[1, 1].errorbar(xpts, upts_im, yerr=(upts_im - rpts_im), ecolor='black', label='$u'+subscript+'(x)$')
    axs[1, 1].set(xlabel='x', ylabel='$Im[u'+subscript+'(x)]$')

    axs[2, 0].plot(xpts, upts_re - rpts_re)
    axs[2, 0].set(xlabel='x', ylabel='$Re[u'+subscript+'(x)-u^N'+subscript+'(x)]$')
    axs[2, 1].plot(xpts, upts_im - rpts_im)
    axs[2, 1].set(xlabel='x', ylabel='$Im[u'+subscript+'(x)-u^N'+subscript+'(x)]$')

    for row in axs:
        for ax in row:
            ax.grid()
            if len(ax.get_legend_handles_labels()[1]) > 0:
                ax.legend()

    if file: plt.savefig(file)