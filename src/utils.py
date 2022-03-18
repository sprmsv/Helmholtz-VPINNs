import numpy as np
from numpy.polynomial import Polynomial
import torch
from typing import Callable, Union

def changeType(x, target='Tensor'):
    if type(x).__name__ != target:
        if target == 'Tensor':
            return torch.tensor(x)

class Legendre_Polynomials:

    def __init__(self, K: int, a: float =-1., b: float =+1.):
        """Calculates the polynomial ceofficients recursively.

        Args:
            K (int): Highest degree of the polynomials.
            a (float): Left boundary of the domain.
            b (float): Right boundary of the domain.

        Raises:
            ValueError: Coefficients become too small to store (round-off error) for degrees more than 40.
        """

        if K > 40:
            raise ValueError('Calculating Legendre polynomials of degree more than 40 requires more accurate data type.')
        c = np.zeros((K + 1, K + 1), dtype=np.float)
        c[0, 0] = 1
        c[1, 0], c[1, 1] = 0, 1

        for k in range(2, K + 1):
            c[k, 0] = - (k - 1) / k * c[k - 2, 0]
            for j in range(1, K + 1):
                c[k, j] = (2 * k - 1) / k * c[k - 1, j - 1] - (k - 1) / k * c[k - 2, j]

        self.K = K
        self.c = c
        self.a = a
        self.b = b

    def poly(self, k: int):
        assert k <= self.K
        return Polynomial(self.c[k],
                        domain=[self.a, self.b], window=[-1, +1])

    def __call__(self):
        """Returns the polynomials in a list.
        """

        polys = []
        for k in range(self.K + 1):
            polys.append(self.poly(k))
        return polys

class derivableFunction:
    def __init__(self, func: Callable, derivs: list =None):
        self.func = func
        self.derivs = derivs if derivs else None

    def deriv(self, i: int):
        assert self.derivs and i <= len(self.derivs)
        if i == 0:
            return self.func
        return self.derivs[i - 1]

    def __call__(self, x: float):
        return self.func(x)

class Finite_Elements:

    def __init__(self, K: int, a: float =-1, b: float =+1, dtype=torch.Tensor):

        self.dtype = dtype
        self.h = (b - a) / K
        self.K = K
        self.a = a
        self.b = b

        if self.dtype is torch.Tensor:
            self.x = torch.linspace(a, b, K + 1)
        elif self.dtype is np.ndarray:
            self.x = np.linspace(a, b, K + 1)

    def __call__(self):
        """Returns the finite elements in a list.
        """

        elements = []
        for k in range(self.K + 1):
            element = derivableFunction(self.phi(k), [self.phi_x(k),])
            elements.append(element)
        return elements


    def phi(self, i: int) -> Callable:
        """Returns the basis functions of the V_N subspace

        Args:
            i (int): Index of the basis function.

        Returns:
            Callable: The basis function.
        """

        xi = self.x[i]

        x_before = self.x[i - 1] if i != 0 else xi
        x_after = self.x[i + 1] if i != self.K else xi

        if self.dtype is torch.Tensor:
            step = lambda x: torch.heaviside(x - x_before, torch.tensor([1.]))\
                - torch.heaviside(x - x_after, torch.tensor([0.]))
            return lambda x: step(x) * (1 - torch.abs((x - xi)) / self.h)

        elif self.dtype is np.ndarray:
            step = lambda x: np.heaviside(x - x_before, 1)\
                - np.heaviside(x - x_after, 0)
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
        x_after = self.x[i + 1] if i != self.K else xi

        if self.dtype is torch.Tensor:
            step = lambda x: torch.heaviside(x - x_before, torch.tensor([1.]))\
                - torch.heaviside(x - x_after, torch.tensor([0.]))
        elif self.dtype is np.ndarray:
            step = lambda x: np.heaviside(x - x_before, 1)\
                - np.heaviside(x - x_after, 0)

        if i == 0:
            return lambda x: step(x) * -(1 / self.h) * (+1)
        elif i == self.K:
            return lambda x: step(x) * -(1 / self.h) * (-1)
        else:
            if self.dtype is torch.Tensor:
                return lambda x: step(x) * -(1 / self.h) * torch.sign(x - xi)
            elif self.dtype is np.ndarray:
                return lambda x: step(x) * -(1 / self.h) * np.sign(x - xi)


    def intphi(self, i: int, j: int) -> float:
        """Calculates int(phi_i * phi_j) over the domain, 0 <= i, j <= N.

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            float: The value of the integral.
        """

        if i == j:
            if i == 0 or i == self.K:
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
            if i == 0 or i == self.K:
                return 1 / self.h
            else:
                return 2 / self.h
        elif abs(i - j) == 1:
            return -1 / self.h
        else:
            return 0