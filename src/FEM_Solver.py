import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union
from quadrature_rules import gauss_lobatto_jacobi_quadrature1D, integrate_1d

class FEM_Helmholtz():
    """ Finite Element Method solver class for solving the 1D Helmholtz equation:
        -u_xx - k^2 * u = f
    in the domain (a, b), with impedance boundary conditions:
        u_x(a) + 1j * k * u(a) = ga
        u_x(b) - 1j * k * u(b) = gb
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
            N_quad (int, optional): Number of quadrature points.
        """

        self.f = f
        self.k = k
        self.a, self.b = a, b
        self.ga, self.gb = ga, gb
        self.N = N
        if not N_quad:
            if self.N * 10 > 1000:
                self.N_quad = 1000
                print(f'Warning: More quadrature points are needed for N={self.N}. The final accuracy might be affected.')
            else:
                self.N_quad = self.N * 10
        else:
            self.N_quad = N_quad

        self.x = np.linspace(a, b, N + 1)
        self.A = np.zeros((N + 1, N + 1), dtype=complex)
        self.d = np.zeros(N + 1, dtype=complex)

        self.c = None
        self.sol = None

        self.roots, self.weights = gauss_lobatto_jacobi_quadrature1D(self.N_quad, a, b)

    def solve(self):
        """Executes the method.
        """

        for i in range(self.N + 1):
            phi_i = self.phi(i)
            phi_x_i = self.phi_x(i)
            self.d[i] = self.rhs(phi_i)
            for j in range(self.N + 1):
                phi_j = self.phi(j)
                phi_x_j = self.phi_x(j)
                self.A[i, j] = self.lhs(phi_j, phi_i, phi_x_j, phi_x_i)

        self.c = np.linalg.solve(self.A, self.d)
        self.sol = lambda x: np.sum(np.array(
            [self.c[i] * self.phi(i)(x) for i in range(self.N + 1)]
            ))

    def phi(self, i: int) -> Callable:
        """Returns the basis functions of the V_N subspace

        Args:
            i (int): Index of the basis function.

        Returns:
            Callable: The basis function.
        """

        xi = self.x[i]
        rng = (self.b - self.a)
        return lambda x: 1 - np.abs((x - xi)) * self.N / rng

    def phi_x(self, i: int) -> Callable:
        """Returns the derivative of the basis function.

        Args:
            i (int): Index of the basis function.

        Returns:
            Callable: The derivative of the basis function
        """

        xi = self.x[i]
        rng = (self.b - self.a)
        slope = self.N / rng
        return lambda x: -slope * np.sign(x - xi)

    def lhs(self, u: Callable, v: Callable, u_x: Callable, v_x: Callable) -> complex:
        """Computes the left-hand-side of the system of the equation:
            int(u_x * v_x) - k^2 * int(u * v) - 1j * k * (u(a) * v(a) + u(b) * v(b))

        Args:
            u (Callable): Basis function.
            v (Callable): Test function.
            u_x (Callable): Derivative of the basis function.
            v_x (Callable): Derivative of the test function.

        Returns:
            complex: The left-hand-side of the equation.
        """

        uv = lambda x: u(x) * v(x)
        uxvx = lambda x: u_x(x) * v_x(x)
        return self.intg(uxvx) - self.k ** 2 * self.intg(uv)\
            - 1j * self.k * (u(self.a) * v(self.a) + u(self.b) * v(self.b))

    def rhs(self, v: Callable) -> complex:
        """Computes the left-hand-side of the system of the equation:
            int(f * v) - ga * v(a) + gb * v(b)

        Args:
            v (Callable): Test function.

        Returns:
            complex: The right-hand-side of the equation.
        """

        fv = lambda x: self.f(x) * v(x)
        return self.intg(fv) - self.ga * v(self.a) + self.gb * v(self.b)

    def intg(self, f: Callable) -> complex:
        """Integrator of the method.

        Args:
            f (Callable): Function to be integrated.

        Returns:
            complex: Integral over the domain (a, b) with N_quad quadrature points.
        """

        y = integrate_1d(f, self.a, self.b, self.weights, self.roots).item()
        return y

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
        return self.sol(x)
