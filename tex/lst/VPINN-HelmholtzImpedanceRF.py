class VPINN_HelmholtzImpedanceRF(VPINN_HelmholtzImpedance):
    def __init__(self, f: Union[Callable, float], k: float, a: float, b: float, ga: complex, gb: complex, *, layers=[1, 10, 10, 10, 2], activation=torch.tanh, penalty=None, quad_N=80, seed=None, cuda=False):

        # Initialize
        assert len(layers) >= 5, "this network only works for D > 2."
        super().__init__(f, k, a, b, ga, gb,
                layers=layers, activation=activation,
                penalty=penalty, quad_N=quad_N, seed=seed, cuda=cuda)
        self.stepact = lambda x: torch.heaviside(-(x-1.), torch.zeros(1)) * torch.relu(x)

    def forward(self, x):
        for i, f in zip(range(self.length), self.lins):
            if i == len(self.lins) - 1:
            # Last layer
                x = f(x)
            elif i == 0:
            # First hidden layer
                x = self.stepact(f(x))
            else:
            # other hidden layers
                x = self.activation(f(x))
        return x