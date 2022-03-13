import numpy as np
from numpy.polynomial import Polynomial
import torch

def changeType(x, target='Tensor'):
    if type(x).__name__ != target:
        if target == 'Tensor':
            return torch.tensor(x)

class Legendre_Polynomials:

    def __init__(self, K: int, a: float =-1, b: float =+1):
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