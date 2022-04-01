import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from solvers import Exact_HelmholtzImpedance, VPINN_HelmholtzImpedance
from testfuncs import Finite_Elements, Legendre_Polynomials
from utils import plot_history, plot_validation

args_activationtype = 'ReLU'
args_testfunctype = 'Finite Elements'
args_dir = None
args_name = None

# Parameters
depth = 1
width = 50
testfuncs = 30
penalty = None
cuda = False
seed = None
dropout_probs = None
init_perfect = False

# Define the parameters of the equation
f = lambda x: 10  # Source function
k = 2. * (np.pi / 2)  # frequency
a, b = -1., +1.  # Domain
ga, gb = 5., 0.  # Values at the boundaries

# Activation function
if args_activationtype == 'ReLU':
    activation = F.relu
elif args_activationtype == 'ReLU Squared':
    activation = lambda x: F.relu(x).pow(2)
    activation.__name__ = 'relu2'
elif args_activationtype == 'CeLU':
    activation = F.celu
elif args_activationtype == 'GeLU':
    activation = F.gelu
else:
    raise ValueError('Activation function not recognized.')

# Test functions
if args_testfunctype == 'Finite Elements':
    testfunctions = Finite_Elements(testfuncs - 1, a, b, dtype=torch.Tensor)
elif args_testfunctype == 'Legendre Polynomials':
    testfunctions = Legendre_Polynomials(testfuncs - 1, a, b)
else:
    raise ValueError('Test function not recognized.')

# Set the directories
file_dir = args_dir if args_dir else './results/VPINN_HelmholtzImpedance/'
experiment_name = args_name if args_name else \
    f'D{depth:03d}N{width:03d}K{testfuncs:03d}-{testfunctions.__name__}-{activation.__name__}'

# Get the exact solution
exact = Exact_HelmholtzImpedance([f(0), 0], k, a, b, ga, gb, source='const')
exact.verify()
u, u_x, u_xx = exact()

# Check that the solution satisfies boundary conditions
assert np.allclose(- u_x(a) - 1j * k * u(a), ga)
assert np.allclose(+ u_x(b) - 1j * k * u(b), gb)
for x in np.linspace(a, b, 100):
    assert np.allclose(- u_xx(x) - k ** 2 * u(x), f(x))

# Model
model = VPINN_HelmholtzImpedance(f=f, k=k, a=a, b=b, ga=ga, gb=gb,
                                layers=[1] + [width for _ in range(depth)] + [2],
                                activation=activation,
                                dropout_probs=dropout_probs,
                                penalty=penalty,
                                quad_N=100,
                                seed=seed,
                                cuda=cuda,
                                )
if cuda: model = model.cuda()

# Initialize close to the solution to check the convergence
if init_perfect:
    points = torch.linspace(a - 1e-06, b, width + 1).float()
    derivs = torch.zeros_like(model.lins[1].weight)
    for i, point, next in zip(range(width), points[:-1], points[1:]):
        derivs[0, i] = .5 * (u_x(point).real + u_x(next).real)
        derivs[1, i] = .5 * (u_x(point).imag + u_x(next).imag)
    steps = derivs.clone()
    for i in range(width - 1):
        steps[0, i + 1] = derivs[0, i + 1] - derivs[0, i]
        steps[1, i + 1] = derivs[1, i + 1] - derivs[1, i]

    model.lins[0].weight = nn.Parameter(torch.ones_like(model.lins[0].weight))
    model.lins[0].bias = nn.Parameter(-1 * points[:-1])
    model.lins[1].weight = nn.Parameter(steps.float())
    model.lins[1].bias = nn.Parameter(torch.tensor([u(a).real, u(a).imag]).float())

stages = []
while True:
    if input('>> Enter "q" to quit: ') == 'q':
        break

    if input('>> Enter "train" to perform the training: ') == 'train':
        # Training parameters
        epochs = int(float(input('Epochs: ')))
        lr = float(input('Learning rate: '))
        # milestones = input('Scheduler milestones: ')
        # milestones = [int(m) for m in milestones.split(',')] if milestones else []
        # gamma = float(input('Scheduler gamma: '))
        milestones, gamma = [], .1
        stages.append({
            'epochs': epochs,
            'lr': lr,
            'milestones': milestones,
            'gamma': gamma,
        })

        # Train
        optimizer = optim.SGD([
        # {'params': model.lins[0].weight, 'lr': 5e-03},
        # {'params': model.lins[0].bias, 'lr': 1e-04},
        {'params': model.lins[1].weight, 'lr': lr},
        {'params': model.lins[1].bias, 'lr': lr},
        ], lr=lr, momentum=.5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=-1)
        model.train_(testfunctions(), epochs, optimizer, scheduler, exact=(u, u_x))

    if input('>> Enter "save" to save the results: ') == 'save':
        model.eval()

        # Save information
        file = file_dir + experiment_name + '-info.json'
        info = {
            'model': {
                'name': model.__class__.__name__,
                'depth': depth,
                'width': width,
                'testfuncs': testfuncs,
                'testfunctions': testfunctions.__name__,
                'activation': activation.__name__,
                'dropout_probs': dropout_probs,
                'penalty': penalty,
            },
            'H1-error': model.history['errors']['tot'][-1] if model.history['errors'] else None,
            'Loss': model.history['losses'][-1] if model.history['losses'] else None,
            'minutes': int(model.time_elapsed // 60),
            'epochs': model.epoch,
            'stages': stages,
            'seed': seed,
            'cuda': cuda,
            'equation': {
                # 'f': f.__name__,
                'k': k,
                'a': a,
                'b': b,
                'ga': ga,
                'gb': gb,
            },
        }
        with open(file, 'w') as f:
            json.dump(info, f, indent=4)

        # Save history
        file = file_dir + experiment_name + '-train_history.json'
        with open(file, 'w') as f:
            json.dump(model.history, f, indent=4)
        file = file_dir + experiment_name + '-train_history.png'
        plot_history(
        model.history,
        file,
        )

        # Plot the parameters
        plt.rcParams['figure.figsize'] = [15, 10]
        fig, axs = plt.subplots(3, 1)
        fig.tight_layout(pad=4.0)
        fig.suptitle(f'Parameters of the model')
        axs[0].scatter(np.arange(1, width + 1), model.lins[0].weight.detach().view(-1).numpy(), label='$w_n$')
        axs[1].scatter(np.arange(1, width + 1), model.lins[0].bias.detach().view(-1).numpy(), label='$b_n$')
        axs[2].scatter(np.arange(1, width + 1), model.lins[1].weight[0].detach().view(-1).numpy(), label='$c^1_n$')
        axs[2].scatter(np.arange(1, width + 1), model.lins[1].weight[1].detach().view(-1).numpy(), label='$c^2_n$')
        axs[2].scatter(0, model.lins[1].bias[0].detach().view(-1).numpy(), label=f'$c^1_0$={model.lins[1].bias[0].item():.2f}', color='black')
        axs[2].scatter(0, model.lins[1].bias[1].detach().view(-1).numpy(), label=f'$c^2_0$={model.lins[1].bias[1].item():.2f}', color='black')
        for ax in axs:
            ax.grid()
            ax.legend()
        file = file_dir + experiment_name + '-weights.png'
        plt.savefig(file)

        # Set the title of the plots
        title = f'\
            k={round(model.k.item()/(np.pi/2))}π/2, \
            ga={model.ga_re.item()}+i{model.ga_im.item()}, \
            gb={model.gb_re.item()}+i{model.gb_im.item()}, \
            \nD={depth}, N={width}, K={testfuncs}\
            \n'

        # Plot the solutions
        xpts = torch.linspace(a, b, 301).float().view(-1, 1)
        upts_re, upts_im = u(xpts).real, u(xpts).imag
        rpts_re, rpts_im = model.deriv(0, xpts)
        with torch.no_grad():
            xpts = xpts.numpy().reshape(-1)
            upts_re = upts_re.numpy().reshape(-1)
            upts_im = upts_im.numpy().reshape(-1)
            rpts_re = rpts_re.numpy().reshape(-1)
            rpts_im = rpts_im.numpy().reshape(-1)

        file = file_dir + experiment_name + '-sol.png'
        plot_validation(
            xpts, (upts_re, upts_im), (rpts_re, rpts_im),
            title=title,
            subscript='',
            file=file,
        )

        # Plot the derivatives
        xpts = torch.linspace(a, b, 301).float().view(-1, 1)
        upts_re, upts_im = u_x(xpts).real, u_x(xpts).imag
        xpts.requires_grad_()
        rpts_re, rpts_im = model.deriv(1, xpts)
        with torch.no_grad():
            xpts = xpts.numpy().reshape(-1)
            upts_re = upts_re.numpy().reshape(-1)
            upts_im = upts_im.numpy().reshape(-1)
            rpts_re = rpts_re.numpy().reshape(-1)
            rpts_im = rpts_im.numpy().reshape(-1)
        file = file_dir + experiment_name + '-der.png'
        plot_validation(
            xpts, (upts_re, upts_im), (rpts_re, rpts_im),
            title=title,
            subscript='_x',
            file=file,
        )

        print(f'>> Results are stored in {file_dir} with the prefix "{experiment_name}-*.*".')

# H1-error
errs = model.H1_error(u, u_x)
print(f'>> H1-error: \t\t{errs[0].item():.2e}')
print(f'>> Solution L2-error: \t{errs[1].item():.2e}')
print(f'>> Derivative L2-error: {errs[2].item():.2e}')
