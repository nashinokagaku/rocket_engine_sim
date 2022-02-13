import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar, shgo, minimize
from scipy.optimize import differential_evolution, dual_annealing

def f(x):
    v = 3 / 7 * np.cos(3 / 2 * x) - np.sin(1 / 3 * x) + 1 / 3 * np.sin(x**2) - 1
    return(v)

# minimize_scaler
res = minimize_scalar(f, bounds=(0, 2*np.pi), method="bounded")
print(res.x)
print(res.fun)

# shgo
res = shgo(f, bounds=[(0, 2*np.pi)])
print(res.x)
print(res.fun)

# minimize
div = 100
x = np.linspace(0, 2*np.pi, div)
y = f(x)
x0 = (2 * np.pi / div) * (np.argmin(y) + 1) # initial point
bnds = ((0, 2 * np.pi),)
res = minimize(f, x0, method='Nelder-Mead', bounds=bnds, tol=1e-6)
print(res.x)
print(res.fun)

# differential_evolution
res = differential_evolution(f, bounds=[(0, 2*np.pi)], disp=False)
print(res.x)
print(res.fun)

# del_annealing
lw = [0]
up = [2 * np.pi]
ret = dual_annealing(f, bounds=list(zip(lw, up)))
print(res.x)
print(res.fun)

# minimize with constraints
# 目的関数
def func(x):
    return x ** 2

# 制約条件式
a = 0
def cons1(x):
    return -a * (x + 1)
def cons2(a):
    return a - 0.5

# 制約条件式が非負となるようにする
cons = (
    {'type': 'ineq', 'fun': cons1},
    {'type': 'eq', 'fun': cons2}
)
x = -10 # 初期値は適当

result = minimize(func, x0=x, constraints=cons, method="SLSQP")
print(result.x[0])
print(result.fun[0])