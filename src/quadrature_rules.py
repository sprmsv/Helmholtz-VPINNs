from scipy.special import roots_jacobi, jacobi, gamma
import numpy as np
import torch


# GAUSS-LOBATTO-JACOBI QUADRATURE RULE
##################################################


def jacobi_evaluation(n, alpha, beta, x):
    '''Recursive generation of the Jacobi polynomial of order n.

    Args:
        n (int): Order of the Jacobi Polynomial, :math:`\ge 0`.
        alpha (float): Alpha parameter of the Jacobi polynomial, :math:`> -1`.
        beta (float): Beta parameter of the Jacobi polynomial, :math:`> -1`.
        x (np.ndarray): Array to evaluate.
    
    Returns:
        y (np.ndarray): A numpy array of the Jacobi polynomial evaluated at x.
        
    Example:
        >>> n, alpha, beta = 1, 0, 0
        >>> x = np.linspace(-1., 1., 5)
        >>> y = jacobi_evaluation(n, alpha, beta, x)
        >>> print(y)
        [-1.  -0.5  0.   0.5  1. ]
        
    '''
    x = np.array(x)
    return jacobi(n, alpha, beta)(x)


def gauss_lobatto_jacobi_weights(q, alpha=0, beta=0):
    '''Returns the roots and weights of the Gauss-Lobatto-Jacobi quadrature rule.

    Args:
        q (int): Number of roots for the quadrature rule, :math:`\ge 1`.
        alpha (float, optional): Alpha parameter of the Jacobi polynomial, :math:`> -1`. Defaults to 0.
        beta (float, optional): Beta parameter of the Jacobi polynomial, :math:`> -1`. Defaults to 0.
            
    Returns:
        roots (np.ndarray): roots of the Jacobi polynomial.
        weights (np.ndarray): weights of the quadrature rule.
    
    Example:

        >>> roots, weights = gauss_lobatto_jacobi_weights(5, 0, 0)
        >>> print(roots, weights)
        [-1.         -0.65465367  0.          0.65465367  1.        ] [0.1        0.54444444 0.71111111 0.54444444 0.1       ]

    '''
    w = []
    x = roots_jacobi(q - 2, alpha + 1, beta + 1)[0]
    if alpha == 0 and beta == 0:
        w = 2 / ((q - 1) * (q) * (jacobi_evaluation(q - 1, 0, 0, x) ** 2))
        wl = 2 / ((q - 1) * (q) * (jacobi_evaluation(q - 1, 0, 0, -1) ** 2))
        wr = 2 / ((q - 1) * (q) * (jacobi_evaluation(q - 1, 0, 0, 1) ** 2))
    else:
        w = 2 ** (alpha + beta + 1) * gamma(alpha + q) * gamma(beta + q) / ((q - 1) * gamma(q)
                                                                            * gamma(alpha + beta + q + 1) * (
                                                                                    jacobi_evaluation(q - 1, alpha, beta,
                                                                                                      x) ** 2))
        wl = (beta + 1) * 2 ** (alpha + beta + 1) * gamma(alpha + q) * gamma(beta + q) / ((q - 1)
                                                                                          * gamma(q) * gamma(
                    alpha + beta + q + 1) * (jacobi_evaluation(q - 1, alpha, beta, -1) ** 2))
        wr = (alpha + 1) * 2 ** (alpha + beta + 1) * gamma(alpha + q) * gamma(beta + q) / ((q - 1)
                                                                                           * gamma(q) * gamma(
                    alpha + beta + q + 1) * (jacobi_evaluation(q - 1, alpha, beta, 1) ** 2))
    w = np.append(w, wr)
    w = np.append(wl, w)
    x = np.append(x, 1)
    x = np.append(-1, x)
    return [x, w]


def gauss_lobatto_jacobi_quadrature1D(num_points, a, b, alpha=0, beta=0):
    """Gives the 1D Gauss-Lobatto-Jacobi quadrature rule.

    Returns the nodes and weights :math:`\{x_i,w_i\}_{i=1}^Q`, such that:
    
    .. math::
        \int_a^b f(x) dx \\approx \sum_{i=1}^Q w_i f(x_i)
        
    The nodes and weights are given by the Gauss-Lobatto-Jacobi quadrature rule. 

    Args:
        num_points (int): Number of nodes of the quadrature rule.
        a (float, torch.Tensor): Start of the integration interval.
        b (float, torch.Tensor): End of the integration interval.
        alpha (float, optional): Alpha parameter of the Jacobi polynomial, :math:`> -1`. Defaults to 0.
        beta (float, optional): Beta parameter of the Jacobi polynomial, :math:`> -1`. Defaults to 0.

    Returns:
        roots (torch.Tensor: roots of the Jacobi polynomial.
        weights (torch.Tensor): weights of the quadrature rule.
        
    Example:

        >>> roots, weights = gauss_lobatto_jacobi_quadrature1D(5, -1, 2)
        >>> print(roots, weights)
        [-1.         -0.48198051  0.5         1.48198051  2.        ] [0.1        0.54444444 0.71111111 0.54444444 0.1       ]
    
    """

    roots, weights = gauss_lobatto_jacobi_weights(num_points, alpha, beta)
    X = (b - a) / 2 * (roots + 1) + a

    if type(X).__name__ != 'Tensor':
        X = torch.tensor(X)

    return X, torch.tensor(weights)


def gauss_lobatto_jacobi_quadrature2D(num_points_per_dim, a, b, c, d,
                                      alpha_x=0, beta_x=0, alpha_y=0, beta_y=0):
    """Gives the 1D Gauss-Lobatto-Jacobi quadrature rule.

    Returns the nodes and weights :math:`\{x_i, w_i\}_{i=1}^Q`, 
    and :math:`\{y_i, \\tilde{w}_i\}_{i=1}^Q`, such that:
    
    .. math::
        \int_a^b \int_c^d f(x,y) dxdy \\approx \sum_{i,j=1}^Q w_i  \\tilde{w}_j f(x_i, y_j)
        
    The nodes and weights are given by the Gauss-Lobatto-Jacobi quadrature rule. 

    Args:
        num_points_per_dim (int): Number of nodes of the quadrature rule (per dimension).
        a (float): Start of the first integration interval.
        b (float): End of the first integration interval.
        c (float): Start of the second integration interval.
        d (float): End of the second integration interval.
        alpha_x (float, optional): Alpha parameter of the first Jacobi polynomial, :math:`> -1`. Defaults to 0.
        beta_x (float, optional): Beta parameter of the first Jacobi polynomial, :math:`> -1`. Defaults to 0.
        alpha_y (float, optional): Alpha parameter of the second Jacobi polynomial, :math:`> -1`. Defaults to 0.
        beta_y (float, optional): Beta parameter of the second Jacobi polynomial, :math:`> -1`. Defaults to 0.

    Returns:
        X (np.ndarray): roots of the Jacobi polynomial for the first dimension.
        Y (np.ndarray): roots of the Jacobi polynomial for the second dimension.
        Weights_X (np.ndarray): weights of the quadrature rule for the first dimension.
        Weights_y (np.ndarray): weights of the quadrature rule for the second dimension.
        
    Example:

        >>> X, Y, weights_X, weights_Y = gauss_lobatto_jacobi_quadrature2D(3, -1, 2, 0, 1)
        >>> print(X, "\\n", Y, "\\n", weights_X, "\\n", weights_Y)
        [[-1.   0.5  2. ]
         [-1.   0.5  2. ]
         [-1.   0.5  2. ]] 
         [[0.  0.  0. ]
         [0.5 0.5 0.5]
         [1.  1.  1. ]] 
         [[0.33333333 1.33333333 0.33333333]
         [0.33333333 1.33333333 0.33333333]
         [0.33333333 1.33333333 0.33333333]] 
         [[0.33333333 0.33333333 0.33333333]
         [1.33333333 1.33333333 1.33333333]
         [0.33333333 0.33333333 0.33333333]]
         
    """

    roots_X, weights_X = gauss_lobatto_jacobi_quadrature1D(
        num_points_per_dim, a, b, alpha_x, beta_x)
    roots_Y, weights_Y = gauss_lobatto_jacobi_quadrature1D(
        num_points_per_dim, c, d, alpha_y, beta_y)

    X, Y = np.meshgrid(roots_X, roots_Y)
    weights_X, weights_Y = np.meshgrid(weights_X, weights_Y)

    return X, Y, weights_X, weights_Y


# INTEGRATION ALGORITHMS
##################################################


def integrate_1d(f, a, b, weights, X=None):
    '''Integrates a function :math:`f(x)` over :math:`[a, b]` using the given roots and weights.
    
    Note that if X is None, then f must be a np.ndarray or torch.Tensor containing the values
    of the function at the nodes.
    
    Args:
        f (Iterable or function handle): if f is an array of the same size as weights,
                                         the integral is directly computed using summation.
                                         Otherwise f is evaluated on the roots before hand.
        a (float): start of the interval.
        b (float): end of the interval.
        weights (Iterable): weights of the quadrature.
        X (Iterable, optional): where the function needs to be evaluated (as accorded to the quadrature rule).  Defaults to None.
    
    Returns:
        integral (float) Value of the integral.
        
    Example:
        >>> a, b = -1, 2
        >>> roots, weights = gauss_lobatto_jacobi_quadrature1D(5, a, b)
        >>> f = lambda x: x**2
        >>> integral = integrate_1d(f, a, b, weights, roots)
        >>> print(integral)
        3.0
    
    '''

    if weights is None:
        raise ValueError("No quadrature weights provided (weights = None)")

    if type(weights).__name__ != "ndarray" and type(weights).__name__ != "Tensor":
        weights = torch.Tensor(weights)

    if X is not None and type(X).__name__ != "ndarray" and type(X).__name__ != "Tensor":
        X = torch.Tensor(X)

    if type(f).__name__ == 'function':
        if X is None:
            raise ValueError(
                "f is a function handle but no values were provided for evaluation (roots = None)")
        values = f(X)
    elif type(f).__name__ != "ndarray" and type(f).__name__ != "Tensor":
        values = torch.Tensor(f)
    else:
        values = f

    if type(values).__name__ == 'Tensor':
        integral = (b - a) * torch.sum(values * weights) / 2
    else:
        integral = (b - a) * np.sum(values * weights) / 2

    return integral


def integrate_2d(f, a, b, c, d, weights_X,
                 weights_Y, X=None, Y=None):
    '''
    Integrates a function :math:`f(x,y)` over :math:`[a, b] \\times [c,d]` using the given roots and weights.
    
    Args:
        f (Iterable or function handle): if f is an array of the same size as weights,
                                         the integral is directly computed using summation.
                                         Otherwise, f is evaluated on the roots beforehand.
        a (float): start of the interval on x.
        b (float): end of the interval on x.
        c (float): start of the interval on y.
        d (float): end of the interval on y.
        weights_X (Iterable): weights for the quadrature.
        weights_Y (Iterable): weights for the quadrature.
        X (Iterable, optional): where the function needs to be evaluated (as accorded to the quadrature rule).  Defaults to None.
        Y (Iterable, optional): where the function needs to be evaluated (as accorded to the quadrature rule). Defaults to None.
    
    Example:
        >>> a, b, c, d = -1, 2, 0, 1
        >>> X, Y, weights_X, weights_Y = gauss_lobatto_jacobi_quadrature2D(3, a, b, c, d)
        >>> f = lambda x, y: x**2 * y**2
        >>> integral = integrate_2d(f, a, b, c, d, weights_X, weights_Y, X, Y)
        >>> print(integral)
        1.0
    
    '''

    if weights_X is None or weights_Y is None:
        raise ValueError("No quadrature weights provided (weights = None)")

    # First we convert everything to numpy arrays:
    if X is not None and type(X).__name__ != "ndarray" and type(X).__name__ != "Tensor":
        X = torch.Tensor(X)
    if Y is not None and type(Y).__name__ != "ndarray" and type(Y).__name != "Tensor":
        Y = torch.Tensor(Y)
    if type(weights_X).__name__ != "ndarray" and type(weights_X).__name__ != "Tensor":
        weights_X = torch.Tensor(weights_X)
    if type(weights_Y).__name__ != "ndarray" and type(weights_Y).__name__ != "Tensor":
        weights_Y = torch.Tensor(weights_Y)

    if type(f).__name__ == 'function':
        if X is None or Y is None:
            raise ValueError(
                "f is a function handle but no values were provided for evaluation (roots = None)")
        values = f(X, Y)
    elif type(f).__name__ != "ndarray" and type(f).__name__ != "Tensor":
        values = torch.Tensor(f)
    else:
        values = f

    if type(values).__name__ == 'Tensor':
        integral = (b - a) * (d - c) * \
                   torch.sum(values * weights_X * weights_Y) / 4
    else:
        integral = (b - a) * (d - c) * np.sum(values * weights_X * weights_Y) / 4

    return integral


if __name__ == "__main__":
    import doctest
    doctest.testmod()
