import numpy as np
from typing import Callable, Union
from quadrature_rules import gauss_lobatto_jacobi_quadrature1D, integrate_1d
import torch
from torch import nn
from numpy.polynomial import Polynomial
from utils import changeType

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

        self.source = source
        self.f = f
        self.k = k
        self.a, self.b = a, b
        self.ga, self.gb = ga, gb
        self.N = N
        self.h = (b - a) / N
        if not N_quad:
            if self.N * 10 > 1000:
                self.N_quad = 1000
                # print(f'Warning: More quadrature points are needed for N={self.N}. The final accuracy might be affected.')
            else:
                self.N_quad = self.N * 10
        else:
            self.N_quad = N_quad

        self.x = np.linspace(a, b, N + 1)
        self.A = np.zeros((N + 1, N + 1), dtype=complex)
        self.d = np.zeros(N + 1, dtype=complex)

        self.c = None
        self.sol = None
        self.der = None

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
            [[self.c[i] * self.phi(i)(x)] for i in range(self.N + 1)]
            ), axis=0)
        self.der = lambda x: np.sum(np.array(
            [[self.c[i] * self.phi_x(i)(x)] for i in range(self.N + 1)]
            ), axis=0)

    def phi(self, i: int) -> Callable:
        """Returns the basis functions of the V_N subspace

        Args:
            i (int): Index of the basis function.

        Returns:
            Callable: The basis function.
        """

        xi = self.x[i]

        x_before = self.x[i - 1] if i != 0 else xi
        x_after = self.x[i + 1] if i != self.N else xi
        step = lambda x: np.heaviside(x - x_before, 1) - np.heaviside(x - x_after, 0)

        return lambda x: step(x) * (1 - np.abs((x - xi)) / self.h)

    def phi_x(self, i: int) -> Callable:
        """Returns the derivative of the basis function.

        Args:
            i (int): Index of the basis function.

        Returns:
            Callable: The derivative of the basis function
        """

        xi = self.x[i]

        x_before = self.x[i - 1] if i != 0 else xi
        x_after = self.x[i + 1] if i != self.N else xi
        step = lambda x: np.heaviside(x - x_before, 1) - np.heaviside(x - x_after, 0)

        if i == 0:
            return lambda x: step(x) * -(1 / self.h) * (+1)
        elif i == self.N:
            return lambda x: step(x) * -(1 / self.h) * (-1)
        else:
            return lambda x: step(x) * -(1 / self.h) * np.sign(x - xi)

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

        phi_i = self.phi(i)
        phi_j = self.phi(j)
        return self.intphi_x(i, j) - self.k ** 2 * self.intphi(i, j)\
            - 1j * self.k * (phi_i(self.a) * phi_j(self.a) + phi_i(self.b) * phi_j(self.b))

    def rhs(self, j: int) -> complex:
        """Computes the right-hand-side of the system of the equations:
            int(f * phi_j) + ga * phi_j(a) + gb * phi_j(b)

        Args:
            j (int): Index of the test function.

        Returns:
            complex: The right-hand-side of the equation.
        """

        phi_j = self.phi(j)

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

    def intphi(self, i: int, j: int) -> float:
        """Calculates int(phi_i * phi_j) over the domain, 0 <= i, j <= N.

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            float: The value of the integral.
        """

        if i == j:
            if i == 0 or i == self.N:
                return self.h / 3
            else:
                return self.h * 2 / 3
        elif abs(i - j) == 1:
            return self.h / 6
        else:
            return 0

    def intphi_x(self, i: int, j: int) -> float:
        """Calculates int(phi_x_i * phi_x_j) over the domain, 0 <= i, j <= N.

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            float: The value of the integral.
        """

        if i == j:
            if i == 0 or i == self.N:
                return 1 / self.h
            else:
                return 2 / self.h
        elif abs(i - j) == 1:
            return -1 / self.h
        else:
            return 0

    def intg(self, f: Callable) -> complex:
        """Integrator of the class.

        Args:
            f (Callable): Function to be integrated.

        Returns:
            complex: Integral over the domain (a, b) with N_quad quadrature points.
        """

        y = integrate_1d(f, self.a, self.b, self.weights, self.roots).item()
        return y

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
# FIXME: Faces an error in integrate_1D()
# TODO: Add calculation of u_x
    def __init__(self, f: Union[Callable, float], f_x: Callable,\
        k: float, a: float, b: float, ga: complex, gb: complex):

        self.f = f
        self.f_x = f_x
        self.k = k
        self.a, self.b = a, b
        self.ga, self.gb = ga, gb

        self.N_quad = 100
        self.roots, self.weights = gauss_lobatto_jacobi_quadrature1D(self.N_quad, a, b)
        self.roots, self.weights = self.roots.numpy(), self.weights.numpy()

    def w(self, x) -> Callable:
        return -np.exp(1j * self.k) / (2j * self.k)\
            * (self.ga * np.exp(1j * self.k * x)
                + self.gb * np.exp(-1j * self.k * x))

    def uG(self, x) -> Callable:
        Hf = lambda s: self.H(x, s) * self.f(s)
        Hf_x = lambda s: self.H(x, s) * self.f_x(s)
        return Hf(self.b) - Hf(self.a) - self.intg(Hf_x, self.a, self.b)

    def H(self, x, s):
        G = lambda t: np.exp(1j * self.k * np.abs(x - t)) / (2j * self.k)
        return self.intg(G, self.a, s)

    def intg(self, f: Callable, a: float =None, b: float=None) -> complex:
        """Integrator of the class.

        Args:
            f (Callable): Function to be integrated.
            a (float): Left boundary.
            b (float): Right boundary.

        Returns:
            complex: Integral over the domain (a, b) with N_quad quadrature points.
        """

        roots, weights = gauss_lobatto_jacobi_quadrature1D(self.N_quad, a, b)
        roots, weights = roots.numpy(), weights.numpy()

        return integrate_1d(f, a, b, weights, roots).item()

    def __call__(self):
        u = lambda x: self.uG(x) + self.w(x)
        return u

def Exact_HelmholtzImpedance_const(f: float, k: float,\
    a: float, b: float, ga: complex, gb: complex):

    uG = lambda x: f / (2 * k ** 2) * np.exp(1j * k) * np.cos(k * x)
    uG_x = lambda x: -f / (2 * k) * np.exp(1j * k) * np.sin(k * x)
    w = lambda x: -np.exp(1j * k) / (2j * k)\
            * (ga * np.exp(1j * k * x)
                + gb * np.exp(-1j * k * x))
    w_x = lambda x: -np.exp(1j * k) / 2\
            * (ga * np.exp(1j * k * x)
                - gb * np.exp(-1j * k * x))

    u = lambda x: uG(x) + w(x)
    u_x = lambda x: uG_x(x) + w_x(x)

    return u, u_x

class VPINN_HelmholtzImpedance(nn.Module):
    def __init__(self, f: Union[Callable, float], k: float, a: float, b: float,
        ga: complex, gb: complex, *, layers=[1, 10, 1], activation=nn.ReLU, dropout_probs=None,
        penalty=None, N_quad=80, seed=None):

        if seed:
            # Ensure reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        super(VPINN_HelmholtzImpedance, self).__init__()
        self.f = f
        self.k = changeType(k, 'Tensor')
        self.a = changeType(a, 'Tensor').view(-1, 1).to(torch.cfloat)
        self.b = changeType(b, 'Tensor').view(-1, 1).to(torch.cfloat)
        self.ga = changeType(ga, 'Tensor').view(-1, 1).to(torch.cfloat)
        self.gb = changeType(gb, 'Tensor').view(-1, 1).to(torch.cfloat)
        self.penalty = penalty

        self.roots, self.weights = gauss_lobatto_jacobi_quadrature1D(N_quad, a, b)
        self.roots = self.roots.view(-1, 1).to(torch.cfloat)
        self.weights = self.weights.view(-1, 1).float()

        if dropout_probs:
            assert len(dropout_probs) == len(layers) - 2

        self.length = len(layers)  # Number of layers
        self.activation = lambda x: activation(x.real) + 1j * activation(x.imag)
        self.lins = nn.ModuleList()  # Linear blocks
        self.drops = nn.ModuleList()  # Dropout
        self.bns = nn.ModuleList()  # Batch-normalization

        # Hidden layers
        for input, output in zip(layers[0:-2], layers[1:-1]):
            self.lins.append(nn.Linear(input, output, bias=True).to(torch.cfloat))
            self.bns.append(nn.BatchNorm1d(output))  # FIXME: .to(torch.cfloat)) ?

        # Output layer
        self.lins.append(nn.Linear(layers[-2], layers[-1], bias=True).to(torch.cfloat))
        self.bns.append(nn.BatchNorm1d(output))  # FIXME: .to(torch.cfloat)) ?

        # Initialize weights
        for lin in self.lins:
            nn.init.xavier_normal_(lin.weight, gain=1.)

        # Assign drop-out probabilities
        if dropout_probs:
            for p in dropout_probs:
                self.drops.append(nn.Dropout(p=p))

    # Forward function without batch-normalization and drop-out
    def forward(self, x):
        for i, f, bn in zip(range(self.length), self.lins, self.bns):
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

    def train_(self, epochs: int, testfunctions: list, optimizer: torch.optim.Optimizer, cuda=False):

        if cuda and not torch.cuda.is_available():
            raise Exception('Cuda is not available.')
        if cuda:
            self.cuda()
            optimizer.cuda()
        self.train()

        losses = []
        K = len(testfunctions)
        for epoch in range(epochs + 1):
            loss = 0
            for v_k in testfunctions:
                loss += torch.abs(self.res(v_k, i=3)) ** 2 / K
            if self.penalty:
                loss_ga = torch.abs(self.ga + self.deriv(1, self.a) + 1j * self(self.a)) ** 2
                loss_gb = torch.abs(self.gb - self.deriv(1, self.b) + 1j * self(self.b)) ** 2
                loss += self.penalty / 2 * (loss_ga + loss_gb)

            losses.append(loss.item())
            if epoch % 100 == 0:
                print(f'Epoch {epoch} / {epochs}: loss = {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def res(self, v_k: Polynomial, i: int):

        u = lambda x: self.deriv(0, x)
        u_x = lambda x: self.deriv(1, x)
        u_xx = lambda x: self.deriv(2, x)

        if i == 1:
            R_k = - self.intg(lambda x: u_xx(x) * v_k(x)) \
                - self.k ** 2 * self.intg(lambda x: u(x) * v_k(x))

        elif i == 2:
            # Consider replacing 0 with [u_x(x) * v_k(x)] at boundaries
            # v_k has "compact support" so it should be zero at the boundaries?
            R_k = + self.intg(lambda x: u_x(x) * v_k.deriv(1)(x)) \
                - self.k ** 2 * self.intg(lambda x: u(x) * v_k(x))\
                - 0

        elif i == 3:
            # Consider replacing 0 with [u_x(x) * v_k(x)] at boundaries
            # v_k has "compact support" so it should be zero at the boundaries?
            R_k = - self.intg(lambda x: u(x) * v_k.deriv(2)(x)) \
                - self.k ** 2 * self.intg(lambda x: u(x) * v_k(x))\
                - 0\
                + (u(self.b) * v_k.deriv(1)(self.b)) - (u(self.a) * v_k.deriv(1)(self.a))

        else:
            raise ValueError(f'{i} is not a valid input for VPINN_HelmholtzImpedance.loss().')

        F_k = self.intg(lambda x: self.f(x) * v_k(x))

        return R_k - F_k

    def deriv(self, n: int, x: torch.tensor, *, func: Callable = None):
        if n not in [0, 1, 2]:
            raise ValueError(f'n = {n} is not a valid derivative.')

        x.requires_grad = True
        if not func:
            f = self(x)
        else:
            f = func(x)
        if n >= 1:
            grad = torch.ones(f.size(), dtype=f.dtype).to(f.device)
            f_x = torch.autograd.grad(f, x, grad_outputs=grad, create_graph=True, allow_unused=False)
        if n >= 2:
            grad = torch.ones(f_x.size(), dtype=f.dtype).to(f.device)
            f_xx = torch.autograd.grad(f_x, x, grad_outputs=None, create_graph=True, allow_unused=False)

        if n == 0:
            return f
        elif n == 1:
            return f_x[0]
        elif n == 2:
            return f_xx[0]

    def intg(self, func: Callable) -> complex:
        """Integrator of the class.

        Args:
            f (Callable): Function to be integrated.

        Returns:
            complex: Integral over the domain (a, b) with N_quad quadrature points.
        """

        integral = (self.b - self.a) * torch.sum(func(self.roots) * self.weights) / 2
        return integral

    def __len__(self):
        return self.length - 2  # Number of hidden layers / depth