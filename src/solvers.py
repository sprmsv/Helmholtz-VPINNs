import numpy as np
from typing import Callable, Union
from quadrature_rules import gauss_lobatto_jacobi_quadrature1D, integrate_1d

class FEM_HelmholtzImpedance():
    """ Finite Element Method solver class for solving the 1D Helmholtz Impendace problem:
        - u_xx - k^2 * u = f
    in the domain (a, b), with impedance boundary conditions:
        - u_x(a) + 1j * k * u(a) = ga
        + u_x(b) - 1j * k * u(b) = gb
    """

    def __init__(self, f: Callable, k: float, a: float, b: float, \
        ga: complex, gb: complex, N: int =50, N_quad: int =None):
        """Initializing the parameters

        Args:
            f (Callable): Source function.
            k (float): Equation coefficient.
            a (float): Left boundary.
            b (float): Right boundary.
            ga (complex): Value of the left boundary condition.
            gb (complex): Value of the right boundary condition.
            N (int, optional): Number of discretization points. Defaults to 50.
            N_quad (int, optional): Number of quadrature points for int(f * phi).
        """

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

        # fv = lambda x: self.f(x) * phi_j(x)
        # intfv = self.intg(fv)
        intfv = self.f(.5 * self.b + .5 * self.a) * self.h
        if j == 0 or j == self.N:
            intfv = intfv / 2

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
    def __init__(self, f: Callable, f_x: Callable,\
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
