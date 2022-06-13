# CS-433 Project 2: Road Segmentation Using U-net

A framework for solving the 1D Helmholtz equation with impedance boundary conditions is provided: 1. Finite element solver; 2. Variational Physics-Informed Neural Network (VPINN) solver. The latter is an implementation of this paper. More details on the methodology are provided in `./tex/Report.pdf`.

## Setup
This setup requires a Default Unix Environment with an installed Python 3.7 or Python 3.8. Use the following command to install all the required libraries:
```bash
pip install -r requirements.txt
```

## Codes and folders
The solvers are implemented in `./src/solvers.py`. Examples of using the FEM solver are provided in `./src/FEM_Order.ipynb` and `./src/FEM_Validation.ipynb` Jupyter notebooks. For the VPINN solvers, `./src/VPINN_train.py` should be called. This could be done by the command:
```bash
python src/VPINN_train.py
```
This command should be executed from the parent folder. The parent directory should be added to `PYTHONPATH` before running this command. In a Unix-based environment, this could be achieved by the following command:
```bash
export PYTHONPATH=${PWD}
```

The following table shows the arguments that control the process.

| Flag                  | Type  | Default           | Description                                                                           |
| --------------------- |-------|-------------------|---------------------------------------------------------------------------------------|
| solver                | str   | 'regular'         | Type of solver. Choices are: 'regular', 'hf', 'rf'.                                   |
| params                | str   | `None`            | Network structure in the format: 'DxxxNxxxKxxx'.                                      |
| freq                  | float | `None`            | Wave number of the equation (k).                                                      |
| act                   | str   | 'relu'            | Activation function. Choices are: 'relu', 'relu2', 'celu', 'gelu', 'sigmoid', 'tanh'. |
| tfs                   | str   | 'Finite Elements' | Type of test functions. Choices are: 'Finite Elements', 'Legendre Polynomials'.       |
| init                  | str   | 'random'          | Initialization method. Choices are: 'random' for random, 'ls' for least-squares.      |
| interactive           | bool  | `False`           | Run in interactive mode.                                                              |
| epochs                | int   | 1000              | Number of training epochs.                                                            |
| lr                    | float | 1e-03             | Training learning rate.                                                               |
| dir                   | str   | `None`            | Directory to save the results.                                                        |
| name                  | str   | `None`            | Name of the experiment.                                                               |
| cuda                  | bool  | `False`           | Uses GPU if True, CPU if False.                                                       |
| seed                  | float | `None`            | Set seed of random number generators.                                                 |
| pen                   | str   | `None`            | Coefficient of the penalty term.                                                      |
| plot_grads            | bool  | `False`           | Stores the gradient wrt to the parameters if `True`.                                  |

### Examples

An example of using the VPINN solvers is provided in `./run.sh`:
```bash
python src/VPINN_train.py\
    --solver regular\
    --params D001N020K020\
    --tfs "Finite Elements"\
    --act tanh\
    --freq 8.\
    --epochs 5000\
    --lr 1e-03\
    --init ls\
    --dir "./results/tmp/"\
    --plot_grads True
```
This command will train a shallow network with 20 hidden layers by testing the weak formulation of the Helmholtz impedance problem against 20 finite element hat functions. The wave number is `k=8`, the equation is solved over the domain $\Omega = (-1,+1)$ with Dirichlet boundary conditions $g_a = 5$ and $g_b = 2$. If needed, the boundary conditions could be changed in `VPINN_train.py`.

## Contributions
This repository corresponds to the code implementations of a Master's semester project entitled *"Variational Physics-Informed Neural Networks For the Helmholtz Impedance Problem"*. This project has been supervised by Prof. Jan S. Hesthaven and Dr. Fernando Henriquez, in the Chair of Computational Mathematics and Simulation Science (MCSS) of École Polytechnique Fédérale de Lausanne (EPFL), Switzerland. The final report of the project is available in `./tex/Report.pdf`.

Author: Sepehr Mousavi ([sepehr.mousavi@epfl.ch](mailto:sepehr.mousavi@epfl.ch))
