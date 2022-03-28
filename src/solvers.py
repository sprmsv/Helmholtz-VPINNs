from typing import Callable, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quadrature_rules import gauss_lobatto_jacobi_quadrature1D, integrate_1d
from utils import changeType
from testfuncs import Finite_Elements


class FEM_HelmholtzImpedance():
    """ Finite Element Method solver class for solving the 1D Helmholtz Impendace problem:
        - u_xx - k^2 * u = f
    in the domain (a, b), with impedance boundary conditions:
        - u_x(a) + 1j * k * u(a) = ga
        + u_x(b) - 1j * k * u(b) = gb
    """

    def __init__(self, f: Union[Callable, float], k: float, a: float, b: float, \
        ga: complex, gb: complex, *, source: str ='const', N: int =50, N_quad: int =None):

        """Initializing the parameters

        Args:
            f (function or float): Source function or the coefficient.
            k (float): Equation coefficient.
            a (float): Left boundary.
            b (float): Right boundary.
            ga (complex): Value of the left boundary condition.
            gb (complex): Value of the right boundary condition.
            source (str): Type of the source function. Valid values are: 'const', 'func'.
            N (int, optional): Number of discretization points. Defaults to 50.
            N_quad (int, optional): Number of quadrature points for int(f * phi).
        """

        # Store equation parameters
        self.source = source
        self.f = f
        self.k = k
        self.a, self.b = a, b
        self.ga, self.gb = ga, gb
        self.N = N
        self.h = (b - a) / N

        # Get basis functions
        self.FE = Finite_Elements(N, a, b, dtype=np.ndarray)
        self.bases = self.FE()
        if not N_quad:
            if self.N * 10 > 1000:
                self.N_quad = 1000
                # print(f'Warning: More quadrature points are needed for N={self.N}. The final accuracy might be affected.')
            else:
                self.N_quad = self.N * 10
        else:
            self.N_quad = N_quad

        # Initialize coefficients
        self.x = np.linspace(a, b, N + 1)
        self.A = np.zeros((N + 1, N + 1), dtype=complex)
        self.d = np.zeros(N + 1, dtype=complex)

        # Initialize solutions
        self.c = None
        self.sol = None
        self.der = None

        # Get the quadrature points
        self.roots, self.weights = gauss_lobatto_jacobi_quadrature1D(self.N_quad, a, b)
        self.roots, self.weights = self.roots.numpy(), self.weights.numpy()

    def solve(self):
        """Executes the method.
        """

        for i in range(self.N + 1):
            self.d[i] = self.rhs(i)
            self.A[i, i] = self.lhs(i, i)
            if i != 0:
                self.A[i, i - 1] = self.lhs(i, i - 1)
            if i != self.N:
                self.A[i, i + 1] = self.lhs(i, i + 1)

        self.c = np.linalg.solve(self.A, self.d)
        self.sol = lambda x: np.sum(np.array(
            [[self.c[i] * self.bases[i](x)] for i in range(self.N + 1)]
            ), axis=0)
        self.der = lambda x: np.sum(np.array(
            [[self.c[i] * self.bases[i].deriv(1)(x)] for i in range(self.N + 1)]
            ), axis=0)

    def lhs(self, i, j) -> complex:
        """Computes the left-hand-side of the system of the equations:
            int(phi_x_i * phi_x_j) - k^2 * int(phi_i * phi_j)
            - 1j * k * (phi_i(a) * phi_j(a) + phi_i(b) * phi_j(b))

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            complex: The left-hand-side of the equation.
        """

        phi_i = self.bases[i]
        phi_j = self.bases[j]
        return self.FE.intphi_x(i, j) - self.k ** 2 * self.FE.intphi(i, j)\
            - 1j * self.k * (phi_i(self.a) * phi_j(self.a) + phi_i(self.b) * phi_j(self.b))

    def rhs(self, j: int) -> complex:
        """Computes the right-hand-side of the system of the equations:
            int(f * phi_j) + ga * phi_j(a) + gb * phi_j(b)

        Args:
            j (int): Index of the test function.

        Returns:
            complex: The right-hand-side of the equation.
        """

        phi_j = self.bases[j]

        if self.source == 'const':
            intfv = self.f * self.h
            if j == 0 or j == self.N:
                intfv = intfv / 2
        elif self.source == 'func':
            fv = lambda x: self.f(x) * phi_j(x)
            intfv = self.intg(fv)
        else:
            raise ValueError(f'{self.source} is not a valid source type.')

        return intfv + self.ga * phi_j(self.a) + self.gb * phi_j(self.b)

    def H1_error(self, u: Callable, u_x: Callable) -> float:
        """Computes the H1 error:
        .. math::
            \\sqrt{||u - u^N||_{L^2}^2 + ||\\frac{du}{dx} - \\frac{du^N}{dx}||_{L^2}^2}

        Args:
            u (Callable): Exact solution
            u_x (Callable): Exact derivative of the solution

        Returns:
            float: H1 error of the solution
            float: L2 norm of the solution
            float: L2 norm of the derivative of the solution
        """

        if not self.sol:
            print(f'Call {self.__class__.__name__}.solve() to find the FEM solution first.')
            return

        u2 = lambda x: abs(u(x) - self.sol(x)) ** 2
        ux2 = lambda x: abs(u_x(x) - self.der(x)) ** 2

        err_u = np.sqrt(self.intg(u2))
        err_u_x = np.sqrt(self.intg(ux2))

        return err_u + err_u_x, err_u, err_u_x

    def intg(self, func: Callable) -> complex:
        """Integrator of the class.

        Args:
            func (Callable): Function to be integrated.

        Returns:
            complex: Integral over the domain (a, b) with N_quad quadrature points.
        """

        # return integrate_1d(f, self.a, self.b, self.weights, self.roots).item()
        return (self.b - self.a) * np.sum(func(self.roots) * self.weights) / 2

    def __call__(self, x: float) -> complex:
        """Returns the solution of the equation.

        Args:
            x (float): Point to evaluate the solution.

        Returns:
            complex: Value of the solution.
        """

        if not self.sol:
            print(f'Call {self.__class__.__name__}.solve() to find the FEM solution first.')
            return
        return self.sol(x), self.der(x)

class Exact_HelmholtzImpedance():

    # FIXME: General case is not right! uG_xx has a f(0) bias!
    def __init__(self, f: tuple, k: float,\
        a: float, b: float, ga: complex, gb: complex,\
        source: str =None):

        self.source = source
        self.f = f[0]
        self.f_x = f[1]
        self.k = k
        self.a, self.b = a, b
        self.ga, self.gb = ga, gb

        self.quad_N = 500
        roots, weights = gauss_lobatto_jacobi_quadrature1D(self.quad_N, a, b)
        roots, weights = roots.numpy(), weights.numpy()
        self.quadpoints = (roots, weights)

    def verify(self):
        u, u_x, u_xx = self()
        assert np.allclose(- u_x(self.a) - 1j * self.k * u(self.a), self.ga)
        assert np.allclose(+ u_x(self.b) - 1j * self.k * u(self.b), self.gb)
        for x in np.linspace(self.a, self.b, 100):
            assert np.allclose(- u_xx(x) - self.k ** 2 * u(x),
                self.f if self.source == 'const' else self.f(x))

    def w(self, x) -> Callable:
        return -np.exp(1j * self.k) / (2j * self.k)\
            * (self.ga * np.exp(1j * self.k * x)
                + self.gb * np.exp(-1j * self.k * x))

    def w_x(self, x) -> Callable:
        return -np.exp(1j * self.k) / 2\
            * (self.ga * np.exp(1j * self.k * x)
                - self.gb * np.exp(-1j * self.k * x))

    def w_xx(self, x) -> Callable:
        return -np.exp(1j * self.k) / 2 * (1j * self.k)\
            * (self.ga * np.exp(1j * self.k * x)
                + self.gb * np.exp(-1j * self.k * x))

    def uG(self, x) -> Callable:
        if self.source == 'const':
            uG = - self.f / (self.k ** 2) * (1 - np.exp(1j * self.k) * np.cos(self.k * x))
        else:
            Hf = lambda s: self.H(x, s) * self.f(s)
            Hf_x = lambda s: self.H(x, s) * self.f_x(s)
            uG = - (Hf(self.b) - Hf(self.a)) - self.intg(Hf_x)
        return uG

    def uG_x(self, x) -> Callable:
        if self.source == 'const':
            uG_x = - self.f / (self.k) * (np.exp(1j * self.k) * np.sin(self.k * x))
        else:
            Hf = lambda s: self.H(x, s, der=1) * self.f(s)
            Hf_x = lambda s: self.H(x, s, der=1) * self.f_x(s)
            uG_x = - (Hf(self.b) - Hf(self.a)) - self.intg(Hf_x)
        return uG_x

    def uG_xx(self, x) -> Callable:
        if self.source == 'const':
            uG_xx = - self.f * (np.exp(1j * self.k) * np.cos(self.k * x))
        else:
            Hf = lambda s: self.H(x, s, der=2) * self.f(s)
            Hf_x = lambda s: self.H(x, s, der=2) * self.f_x(s)
            uG_xx = - (Hf(self.b) - Hf(self.a)) - self.intg(Hf_x)
        return uG_xx

    def H(self, x, s, der=0):
        if np.allclose(self.a, s):
            return 0

        if der == 0:
            G = lambda t: .5 / (1j * self.k) * np.exp(1j * self.k * np.abs(x - t))
        elif der == 1:
            G = lambda t: .5 * np.sign(x - t) * np.exp(1j * self.k * np.abs(x - t))
        elif der == 2:
            G = lambda t: .5 * (1j * self.k) * np.exp(1j * self.k * np.abs(x - t))

        roots, weights = gauss_lobatto_jacobi_quadrature1D(self.quad_N, self.a, s)
        roots, weights = roots.numpy(), weights.numpy()
        return self.intg(G, (roots, weights))

    def intg(self, func: Callable, quadpoints: tuple =None) -> complex:
        """Integrator of the class.

        Args:
            f (Callable): Function to be integrated.
            quadpoints (tuple): Quadrature points (roots, weights). Defaults to None.

        Returns:
            complex: Integral over the domain (a, b) with N_quad quadrature points.
        """

        if not quadpoints: quadpoints = self.quadpoints
        roots, weights = quadpoints

        return (roots[-1] - roots[0]) * np.sum(func(roots) * weights) / 2

    def __call__(self):
        u = lambda x: self.uG(x) + self.w(x)
        u_x = lambda x: self.uG_x(x) + self.w_x(x)
        u_xx = lambda x: self.uG_xx(x) + self.w_xx(x)
        return u, u_x, u_xx

class VPINN_HelmholtzImpedance(nn.Module):

    def __init__(self, f: Union[Callable, float], k: float, a: float, b: float,
        ga: complex, gb: complex, *, layers=[1, 10, 2], activation=F.relu, dropout_probs=None,
        res_id: int =2, penalty=None, quad_N=80, seed=None, cuda=False):

        # Ensure reproducibility
        if seed:
            torch.manual_seed(seed)
            # torch.backends.cudnn.benchmark = True
            # torch.use_deterministic_algorithms(True)

        # Check the validty of the inputs
        if dropout_probs:
            assert len(dropout_probs) == len(layers) - 2
        assert layers[-1] == 2

        # Initialize
        super().__init__()
        self.activation = activation
        self.res_id = res_id
        self.cuda_ = cuda
        self.quad_N = quad_N
        self.history = {
            'epochs': [],
            'losses': [],
            'errors': None,
        }
        self.e = 0

        # Store equation parameters
        self.f = f
        self.k = changeType(k, 'Tensor').float()
        self.penalty = changeType(penalty, 'Tensor').float() if penalty else None
        self.a = changeType(a, 'Tensor').float().view(-1, 1).requires_grad_()
        self.b = changeType(b, 'Tensor').float().view(-1, 1).requires_grad_()
        self.ga_re = changeType(ga.real, 'Tensor').float().view(-1, 1)
        self.ga_im = changeType(ga.imag, 'Tensor').float().view(-1, 1)
        self.gb_re = changeType(gb.real, 'Tensor').float().view(-1, 1)
        self.gb_im = changeType(gb.imag, 'Tensor').float().view(-1, 1)

        # Store quadrature points
        roots, weights = gauss_lobatto_jacobi_quadrature1D(quad_N, a, b)  # FIXME: Change to quad_N for Legendre polynomials!!
        roots = roots.float().view(-1, 1).requires_grad_()
        weights = weights.float().view(-1, 1)
        if cuda:
            roots, weights = roots.cuda(), weights.cuda()
        self.quadpoints = (roots, weights)

        # Move variables to CUDA
        if cuda:
            if not torch.cuda.is_available(): raise Exception('Cuda is not available.')
            self.k = self.k.cuda()
            self.penalty = self.penalty.cuda()
            self.a = self.a.cuda()
            self.b = self.b.cuda()
            self.ga_re = self.ga_re.cuda()
            self.ga_im = self.ga_im.cuda()
            self.gb_re = self.gb_re.cuda()
            self.gb_im = self.gb_im.cuda()

        # Define modules
        self.length = len(layers)  # Number of layers
        self.lins = nn.ModuleList()  # Linear blocks
        self.drops = nn.ModuleList()  # Dropout
        # self.bns = nn.ModuleList()  # Batch-normalization

        # Define the hidden layers
        for input, output in zip(layers[0:-2], layers[1:-1]):
            self.lins.append(nn.Linear(input, output, bias=True))
            # self.bns.append(nn.BatchNorm1d(output))

        # Define the output layer
        self.lins.append(nn.Linear(layers[-2], layers[-1], bias=True))
        # self.bns.append(nn.BatchNorm1d(output))

        # Assign drop-out probabilities
        if dropout_probs:
            for p in dropout_probs:
                self.drops.append(nn.Dropout(p=p))

        # Initialize weights and biases
        for lin in self.lins[:-1]:
            nn.init.ones_(lin.weight)
            # nn.init.xavier_normal_(lin.weight, gain=nn.init.calculate_gain('relu'))
            # nn.init.normal_(lin.weight, mean=1., std=.5)
            lin.bias = nn.Parameter(-1 * torch.linspace(a - .005 * (b - a), b, layers[1] + 1).float()[:-1])
            # nn.init.uniform_(lin.bias, a=a, b=b)
        nn.init.xavier_normal_(self.lins[-1].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.lins[-1].bias)

    def forward(self, x):
        # for i, f, bn in zip(range(self.length), self.lins, self.bns):
        for i, f in zip(range(self.length), self.lins):
            if i == len(self.lins) - 1:
            # Last layer
                x = f(x)
                # x = bn(x)  # Batch-normalization
            else:
            # Hidden layers
                x = f(x)
                # x = bn(x)  # Batch-normalization
                x = self.activation(x)
                # x = self.drops[i - 1](x)  # Drop-out
        return x

    def train_(self, testfunctions: list, epochs: int, optimizer, scheduler=None,
    exact: tuple =None):

        if exact and not self.history['errors']:
            self.history['errors'] = {'tot': [], 'sol': [], 'der': []}
        self.train()
        K = len(testfunctions)

        # Define quadrature points for each test function
        for v_k in testfunctions:
            if v_k.domain_:
                xl = v_k.domain_[0]
                xr = v_k.domain_[2]
                roots, weights = gauss_lobatto_jacobi_quadrature1D(self.quad_N, xl, xr)
                roots = roots.float().view(-1, 1).requires_grad_()
                weights = weights.float().view(-1, 1)
                if self.cuda_:
                    roots, weights = roots.cuda(), weights.cuda()
                v_k.quadpoints = (roots, weights)
            else:
                v_k.quadpoints = self.quadpoints

        epochs += self.e
        for e in range(self.e, epochs + 1):
            loss = 0

            for v_k in testfunctions:
                loss += self.loss_v(v_k, i=self.res_id, quadpoints=v_k.quadpoints) / K

            if self.penalty:
                u_re = lambda x: self.deriv(0, x)[0]
                u_im = lambda x: self.deriv(0, x)[1]
                u_x_re = lambda x: self.deriv(1, x)[0]
                u_x_im = lambda x: self.deriv(1, x)[1]

                loss_ga_re = self.ga_re + u_x_re(self.a) - self.k * u_im(self.a)
                loss_ga_im = self.ga_im + u_x_im(self.a) + self.k * u_re(self.a)
                loss_gb_re = self.gb_re + u_x_re(self.b) - self.k * u_im(self.b)
                loss_gb_im = self.gb_im + u_x_im(self.b) + self.k * u_re(self.b)
                pen = self.penalty / 2 * (loss_ga_re.pow(2) + loss_ga_im.pow(2)
                                        + loss_gb_re.pow(2) + loss_gb_im.pow(2))
                loss += pen

            if e % 1 == 0:
                # Store the loss and the error
                self.history['epochs'].append(e)
                self.history['losses'].append(loss.item())
                if exact:
                    error = self.H1_error(exact[0], exact[1])
                    self.history['errors']['tot'].append(error[0].item())
                    self.history['errors']['sol'].append(error[1].item())
                    self.history['errors']['der'].append(error[2].item())
                # Print the loss and the error
                if exact:
                    print(f'Epoch {e:06d} / {epochs}: loss = {loss.item():.3e}, H1-error = {error[0].item():.3e}')
                else:
                    print(f'Epoch {e:06d} / {epochs}: loss = {loss.item():.3e}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.e += 1

    def loss_v(self, v_k: Callable, i: int, quadpoints: tuple):

        F_k_re = self.intg(lambda x: self.f(x) * v_k(x), quadpoints)
        F_k_im = 0  # In case of complex source function

        u_re = lambda x: self.deriv(0, x)[0]
        u_im = lambda x: self.deriv(0, x)[1]

        if i == 1:
            u_xx_re = lambda x: self.deriv(2, x)[0]
            u_xx_im = lambda x: self.deriv(2, x)[1]

            R_k_re = - self.intg(lambda x: u_xx_re(x) * v_k(x), quadpoints) \
                - self.k.pow(2) * self.intg(lambda x: u_re(x) * v_k(x), quadpoints)

            R_k_im = - self.intg(lambda x: u_xx_im(x) * v_k(x)) \
                - self.k.pow(2) * self.intg(lambda x: u_im(x) * v_k(x), quadpoints)

        elif i == 2:
            u_x_re = lambda x: self.deriv(1, x)[0]
            u_x_im = lambda x: self.deriv(1, x)[1]

            R_k_re = self.intg(lambda x: u_x_re(x) * v_k.deriv(1)(x), quadpoints) \
                - self.k.pow(2) * self.intg(lambda x: u_re(x) * v_k(x), quadpoints)\
                - (self.ga_re - self.k * u_im(self.a)) * v_k(self.a)\
                - (self.gb_re - self.k * u_im(self.b)) * v_k(self.b)

            R_k_im = self.intg(lambda x: u_x_im(x) * v_k.deriv(1)(x), quadpoints) \
                - self.k.pow(2) * self.intg(lambda x: u_im(x) * v_k(x), quadpoints)\
                - (self.ga_im + self.k * u_re(self.a)) * v_k(self.a)\
                - (self.gb_im + self.k * u_re(self.b)) * v_k(self.b)

        elif i == 3:
            R_k_re = - self.intg(lambda x: u_re(x) * v_k.deriv(2)(x), quadpoints) \
                - self.k.pow(2) * self.intg(lambda x: u_re(x) * v_k(x), quadpoints)\
                - (self.ga_re - self.k * u_im(self.a)) * v_k(self.a)\
                - (self.gb_re - self.k * u_im(self.b)) * v_k(self.b)\
                + (u_re(self.b) * v_k.deriv(1)(self.b)) - (u_re(self.a) * v_k.deriv(1)(self.a))

            R_k_im = - self.intg(lambda x: u_im(x) * v_k.deriv(2)(x), quadpoints) \
                - self.k.pow(2) * self.intg(lambda x: u_im(x) * v_k(x), quadpoints)\
                - (self.ga_im + self.k * u_re(self.a)) * v_k(self.a)\
                - (self.gb_im + self.k * u_re(self.b)) * v_k(self.b)\
                + (u_im(self.b) * v_k.deriv(1)(self.b)) - (u_im(self.a) * v_k.deriv(1)(self.a))

        else:
            raise ValueError(f'{i} is not a valid input for VPINN_HelmholtzImpedance.loss().')

        return (R_k_re - F_k_re).pow(2) + (R_k_im - F_k_im).pow(2)

    def deriv(self, n: int, x: torch.tensor):
        if n not in [0, 1, 2]:
            raise ValueError(f'n = {n} is not a valid derivative.')

        f = self(x)
        f_re = f[:, 0]
        f_im = f[:, 1]
        if n >= 1:
            grad = torch.ones(f_re.size(), dtype=f.dtype, device=f.device)
            f_x_re = torch.autograd.grad(f_re, x, grad_outputs=grad, create_graph=True, allow_unused=False)[0]
            grad = torch.ones(f_im.size(), dtype=f.dtype, device=f.device)
            f_x_im = torch.autograd.grad(f_im, x, grad_outputs=grad, create_graph=True, allow_unused=False)[0]
        if n >= 2:
            grad = torch.ones(f_x_re.size(), dtype=f.dtype, device=f.device)
            f_xx_re = torch.autograd.grad(f_x_re, x, grad_outputs=grad, create_graph=True, allow_unused=False)[0]
            grad = torch.ones(f_x_im.size(), dtype=f.dtype, device=f.device)
            f_xx_im = torch.autograd.grad(f_x_im, x, grad_outputs=grad, create_graph=True, allow_unused=False)[0]

        if n == 0:
            return f_re.view(-1, 1), f_im.view(-1, 1)
        elif n == 1:
            return f_x_re, f_x_im
        elif n == 2:
            return f_xx_re, f_xx_im

    def H1_error(self, u: Callable, u_x: Callable) -> float:
        """Computes the H1 error:
        .. math::
            \\sqrt{||u - u^N||_{L^2}^2 + ||\\frac{du}{dx} - \\frac{du^N}{dx}||_{L^2}^2}

        Args:
            u (Callable): Exact solution
            u_x (Callable): Exact derivative of the solution

        Returns:
            float: H1 error of the solution
            float: L2 norm of the solution
            float: L2 norm of the derivative of the solution
        """
        def sol(x):
            sol = u(x.detach().view(-1).numpy())
            return torch.Tensor([sol.real, sol.imag]).T
        def der(x):
            der = u_x(x.detach().view(-1).numpy())
            return torch.Tensor([der.real, der.imag]).T

        u2 = lambda x: (sol(x) - self(x)).pow(2).sum(axis=1)
        ux2_re = lambda x: (der(x)[:, 0] - self.deriv(1, x)[0].view(-1)).pow(2)
        ux2_im = lambda x: (der(x)[:, 1] - self.deriv(1, x)[1].view(-1)).pow(2)
        ux2 = lambda x: ux2_re(x) + ux2_im(x)

        err_u = torch.sqrt(self.intg(u2))
        err_u_x = torch.sqrt(self.intg(ux2))

        return err_u + err_u_x, err_u, err_u_x

    def intg(self, func: Callable, quadpoints: tuple =None) -> complex:
        """Integrator of the class.

        Args:
            f (Callable): Function to be integrated.
            quadpoints (tuple): Quadrature points (roots, weights). Defaults to None.

        Returns:
            complex: Integral over the domain (a, b) with N_quad quadrature points.
        """

        if not quadpoints: quadpoints = self.quadpoints
        roots, weights = quadpoints

        return (roots[-1] - roots[0]) * torch.sum(func(roots) * weights) / 2

    def __len__(self):
        return self.length - 2  # Number of hidden layers / depth
