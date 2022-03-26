import matplotlib.pyplot as plt
import torch


def changeType(x, target='Tensor'):
    if type(x).__name__ != target:
        if target == 'Tensor':
            return torch.tensor(x)

def plot_train_process(losses, errors):
    plt.rcParams['figure.figsize'] = [15, 10]
    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=4.0)
    fig.suptitle(f'Parameters of the model')

    axs[0].plot(range(len(losses)), losses, label='Total Loss')
    axs[1].plot(range(len(losses)), [error[0] for error in errors], label='Solution H1-error')
    # axs[1].plot(range(len(losses)), [error[1] for error in errors], label='Solution L2-error')
    axs[1].plot(range(len(losses)), [error[2] for error in errors], label='Derivative L2-error')

    axs[0].set(xscale='linear', yscale='log', xlabel='Epoch', ylabel='Loss')
    axs[1].set(xscale='linear', yscale='log', xlabel='Epoch', ylabel='Error')

    for ax in axs:
        ax.grid(which='both')
        ax.legend()
