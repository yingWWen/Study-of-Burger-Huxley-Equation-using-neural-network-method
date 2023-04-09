from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp  # for reference
import autograd.numpy.random as npr

import autograd.numpy as np
from Study_of_Burger_Huxley_Equation_using_neural_network_method.lieneuralsolver import NNSolver
# import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams['legend.fontsize'] = 10

plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 10





def f(t, y):
    # Burgers Huxley equation
    c = -3/2


    X = y[0]
    Y = y[1]
#    Z = y[2]

    return [Y, -c*Y-X*Y+X+(X**3)-2*(X**2)]


t = np.linspace(-5, 3, 100).reshape(-1, 1)

y0_list = [0.924142, -0.0350519]
sol = solve_ivp(f, [t.min(), t.max()], y0_list,
                t_eval=t.ravel(), method='Radau', rtol=1e-5)
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$u$', fontsize=10)
plt.plot(sol.t, sol.y[0],  label='$u_1$')
plt.plot(sol.t, sol.y[1],   label='$u_2$')
plt.legend()
plt.show()

nn = NNSolver(f, t, y0_list, n_hidden=30)
print(nn)
nn.reset_weights()
nn.train(maxiter=3000, iprint=200)
nn.loss_func(nn.params_list)
print(nn.loss_func(nn.params_list))
print(nn.params_list[0])

nn.plot_loss()
y_pred_list, dydt_pred_list = nn.predict()


plt.figure(figsize= (8, 6))
plt.xlabel(r'$\xi$', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.plot(t, y_pred_list[0], 'o', label='Neural solution - $\hat{u}_1$')
plt.plot(sol.t, sol.y[0], label='Exact solution - $u$')
plt.xlabel(r'$\xi$', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.legend(loc = "best")
plt.show()

np.mean([sqrt(mean_squared_error(sol.y[i], y_pred_list[i])) for i in range(2)])
print(np.mean([sqrt(mean_squared_error(sol.y[i], y_pred_list[i])) for i in range(2)]))


t_test = np.linspace(-5, 3.3, 120).reshape(-1,1)
sol = solve_ivp(f, [t_test.min(), t_test.max()], y0_list, method='Radau', rtol=1e-5)
y_pred_list, dydt_pred_list = nn.predict(t=t_test)
plt.figure(figsize=(8, 6))
plt.xlabel(r'$\xi$', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.plot(t_test, y_pred_list[0], 'o', label='Neural solution - $\hat{u}_1$')
plt.plot(sol.t, sol.y[0], label='Exact solution - $u$')
plt.xlabel(r'$\xi$', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.legend(loc = "best")
plt.show()