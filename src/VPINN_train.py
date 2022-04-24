import os
import json
import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from solvers import Exact_HelmholtzImpedance, VPINN_HelmholtzImpedance
from testfuncs import Finite_Elements, Legendre_Polynomials
from utils import plot_history, plot_validation

parser = argparse.ArgumentParser()

parser.add_argument('--params', type=str,
                    help='Network structure: DxxxNxxxKxxx', dest='params', required=True)

parser.add_argument('--act', type=str, default='relu',
                    choices=['relu', 'relu2', 'celu', 'gelu', 'sigmoid', 'tanh'],
                    help='Activation function', dest='activation_type', required=False)

parser.add_argument('--freq', type=float, default=None,
                    help='Frequency of the equation (k)', dest='freq', required=False)

parser.add_argument('--tfs', type=str, default='Finite Elements',
                    choices=['Finite Elements', 'Legendre Polynomials'],
                    help='Test functions', dest='testfuncs_type', required=False)

parser.add_argument('--dir', type=str, default=None,
                    help='Directory to save the results', dest='dir', required=False)

parser.add_argument('--name', type=str, default=None,
                    help='Experiment name', dest='name', required=False)

parser.add_argument('--cuda', type=ast.literal_eval, default=False,
                    help='Use GPU', dest='cuda', required=False)

parser.add_argument('--seed', type=int, default=None,
                    help='Random seed', dest='seed', required=False)

parser.add_argument('--pen', type=float, default=None,
                    help='Penalty term coefficient', dest='penalty', required=False)

parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs', dest='epochs', required=False)

parser.add_argument('--lr', type=float, default=1e-03,
                    help='Learning rate', dest='lr', required=False)

parser.add_argument('--init_optimal', type=ast.literal_eval, default=False,
                    help='Optimal initialization', dest='init_optimal', required=False)

parser.add_argument('--plot_grads', type=ast.literal_eval, default=False,
                    help='Save the plot of the gradients', dest='plot_grads', required=False)

parser.add_argument('--interactive', type=ast.literal_eval, default=False,
                    help='Run in interactive mode', dest='interactive', required=False)


def main(args):

    # Parameters
    depth = int(args.params[1:4])
    width = int(args.params[5:8])
    testfuncs = int(args.params[9:12])
    dropout_probs = None

    # SOURCE FUNCTION
    f = lambda x: 5
    f_x = lambda x: 0
    # f = lambda x: x ** (-.25)
    # f_x = lambda x: -.25 * x ** (-1.25)

    # FREQUENCY
    k = args.freq if args.freq else 6. * (np.pi / 2)

    # BOUNDARY
    a, b = -1., +1.
    ga, gb = 5., 2.

    # Activation function
    if args.activation_type == 'relu':
        activation = F.relu
    elif args.activation_type == 'relu2':
        activation = lambda x: F.relu(x).pow(2)
        activation.__name__ = 'relu2'
    elif args.activation_type == 'sigmoid':
        activation = F.sigmoid
    elif args.activation_type == 'tanh':
        activation = F.tanh
    elif args.activation_type == 'celu':
        activation = F.celu
    elif args.activation_type == 'gelu':
        activation = F.gelu
    else:
        raise ValueError('Activation function not recognized.')

    # Test functions
    if args.testfuncs_type == 'Finite Elements':
        testfunctions = Finite_Elements(testfuncs - 1, a, b, dtype=torch.Tensor
            , device=torch.device('cuda') if args.cuda else torch.device('cpu'))
    elif args.testfuncs_type == 'Legendre Polynomials':
        testfunctions = Legendre_Polynomials(testfuncs - 1, a, b)
    else:
        raise ValueError('Test function not recognized.')

    # Set the directories
    file_dir = args.dir if args.dir else './results/VPINN_HelmholtzImpedance/'
    experiment_name = args.name if args.name else \
        f'D{depth:03d}N{width:03d}K{testfuncs:03d}-{testfunctions.__name__}-{activation.__name__}'

    # Get the exact solution
    exact = Exact_HelmholtzImpedance([f(0), 0], k, a, b, ga, gb, source='const')
    # exact = Exact_HelmholtzImpedance([f, f_x], k, a, b, ga, gb, source=None)
    exact.verify()
    u, u_x = exact()

    # Model
    model = VPINN_HelmholtzImpedance(f=f, k=k, a=a, b=b, ga=ga, gb=gb,
                                    layers=[1] + [width for _ in range(depth)] + [2],
                                    activation=activation,
                                    dropout_probs=dropout_probs,
                                    penalty=args.penalty,
                                    quad_N=100,
                                    seed=args.seed,
                                    cuda=args.cuda,
                                    )
    if args.cuda: model = model.cuda()

    # Initialize close to the solution to check the convergence
    if args.init_optimal:
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
        if args.interactive and input('>> Enter "q" to quit: ') == 'q':
            break

        if not args.interactive or input('>> Enter "train" to perform the training: ') == 'train':
            # Training parameters
            if args.interactive:
                epochs = int(float(input('Epochs: ')))
                lr = float(input('Learning rate: '))
                # milestones = input('Scheduler milestones: ')
                # milestones = [int(m) for m in milestones.split(',')] if milestones else []
                # gamma = float(input('Scheduler gamma: '))
            else:
                epochs = args.epochs
                lr = args.lr
                # milestones = args.milestones
                # gamma = args.gamma
            milestones, gamma = [], .1
            stages.append({
                'epochs': epochs,
                'lr': lr,
                'milestones': milestones,
                'gamma': gamma,
            })

            # Train
            optimizer_params = []
            for lin in model.lins[:-1]:
                # optimizer_params.append({'params': lin.weight, 'lr': lr})
                optimizer_params.append({'params': lin.bias, 'lr': lr})
            optimizer_params.append({'params': model.lins[-1].weight, 'lr': lr})
            optimizer_params.append({'params': model.lins[-1].bias, 'lr': lr})

            optimizer = optim.Adam(
                optimizer_params,
                # momentum=.9,
                )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=-1)
            model.train_(testfunctions(), epochs, optimizer, scheduler, exact=(u, u_x))

        if not args.interactive or input('>> Enter "save" to save the results: ') == 'save':
            model.eval()

            # Create the directory
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

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
                    'penalty': args.penalty,
                },
                'H1-error': model.history['errors']['tot'][-1] if model.history['errors'] else None,
                'Loss': model.history['losses'][-1] if model.history['losses'] else None,
                'minutes': int(model.time_elapsed // 60),
                'epochs': model.epoch,
                'stages': stages,
                'seed': args.seed,
                'device': model.device.type,
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
            axs[0].scatter(np.arange(1, width + 1), model.lins[0].weight.detach().view(-1).cpu().numpy(), label='$w_n$')
            axs[1].scatter(np.arange(1, width + 1), model.lins[0].bias.detach().view(-1).cpu().numpy(), label='$b_n$')
            axs[2].scatter(np.arange(1, width + 1), model.lins[1].weight[0].detach().view(-1).cpu().numpy(), label='$c^1_n$')
            axs[2].scatter(np.arange(1, width + 1), model.lins[1].weight[1].detach().view(-1).cpu().numpy(), label='$c^2_n$')
            axs[2].scatter(0, model.lins[1].bias[0].detach().view(-1).cpu().numpy(), label=f'$c^1_0$={model.lins[1].bias[0].item():.2f}', color='black')
            axs[2].scatter(0, model.lins[1].bias[1].detach().view(-1).cpu().numpy(), label=f'$c^2_0$={model.lins[1].bias[1].item():.2f}', color='black')
            for ax in axs:
                ax.grid()
                ax.legend()
            file = file_dir + experiment_name + '-params.png'
            plt.savefig(file)

            # Plot the gradients
            if args.plot_grads and model.epoch:
                plt.rcParams['figure.figsize'] = [15, 10]
                fig, axs = plt.subplots(3, 1)
                fig.tight_layout(pad=4.0)
                fig.suptitle(f'Gradient of the loss w.r.t the parameters of the model')
                axs[0].scatter(np.arange(1, width + 1), model.lins[0].weight.grad.detach().view(-1).cpu().numpy(), label='$w_n$')
                axs[1].scatter(np.arange(1, width + 1), model.lins[0].bias.grad.detach().view(-1).cpu().numpy(), label='$b_n$')
                axs[2].scatter(np.arange(1, width + 1), model.lins[1].weight.grad[0].detach().view(-1).cpu().numpy(), label='$c^1_n$')
                axs[2].scatter(np.arange(1, width + 1), model.lins[1].weight.grad[1].detach().view(-1).cpu().numpy(), label='$c^2_n$')
                axs[2].scatter(0, model.lins[1].bias.grad[0].detach().view(-1).cpu().numpy(), label=f'$c^1_0$={model.lins[1].bias.grad[0].item():.2f}', color='black')
                axs[2].scatter(0, model.lins[1].bias.grad[1].detach().view(-1).cpu().numpy(), label=f'$c^2_0$={model.lins[1].bias.grad[1].item():.2f}', color='black')
                for ax in axs:
                    ax.grid()
                    ax.legend()
                file = file_dir + experiment_name + '-grads.png'
                plt.savefig(file)

            # Set the title of the plots
            title = f'\
                k={model.k.item():.1f}, \
                ga={model.ga_re.item()}+i{model.ga_im.item()}, \
                gb={model.gb_re.item()}+i{model.gb_im.item()}, \
                \nD={depth}, N={width}, K={testfuncs}\
                \n'

            # Plot the solutions
            xpts = torch.linspace(a, b, 301).float().view(-1, 1)
            upts_re, upts_im = u(xpts).real, u(xpts).imag
            rpts_re, rpts_im = model.deriv(0, xpts.to(model.device))
            with torch.no_grad():
                xpts = xpts.numpy().reshape(-1)
                upts_re = upts_re.numpy().reshape(-1)
                upts_im = upts_im.numpy().reshape(-1)
                rpts_re = rpts_re.cpu().numpy().reshape(-1)
                rpts_im = rpts_im.cpu().numpy().reshape(-1)

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
            rpts_re, rpts_im = model.deriv(1, xpts.to(model.device))
            with torch.no_grad():
                xpts = xpts.numpy().reshape(-1)
                upts_re = upts_re.numpy().reshape(-1)
                upts_im = upts_im.numpy().reshape(-1)
                rpts_re = rpts_re.cpu().numpy().reshape(-1)
                rpts_im = rpts_im.cpu().numpy().reshape(-1)
            file = file_dir + experiment_name + '-der.png'
            plot_validation(
                xpts, (upts_re, upts_im), (rpts_re, rpts_im),
                title=title,
                subscript='_x',
                file=file,
            )

            print(f'>> Results are stored in {file_dir} with the prefix "{experiment_name}-*.*".')

        if not args.interactive:
            break

    # H1-error
    errs = model.H1_error(u, u_x)
    print(f'>> H1-error: \t\t{errs[0].item():.2e}')
    print(f'>> Solution L2-error: \t{errs[1].item():.2e}')
    print(f'>> Derivative L2-error: {errs[2].item():.2e}')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise ValueError('CUDA is not available.')
    main(args)
