# coding: utf-8

# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit

## Exercise 1

df = pd.read_html('https://fred.stlouisfed.org/series/GDPCA')
type(df)

df = np.asarray(df[:][0])
year = df[:,0]           # First column contains the years (+ string 'View all')

i= 0
n=len(df[:])-2   # Subtract number of nan-rows (first and last)
ret = np.empty([n,1]) # Declare array of returns as nans of shape (n,1)
logret = np.empty([n,1]) # Same for logreturns

# Filling empty arrays with computed returns:   

for i in range (0,n):
    ret[i] = df[i][1]/df[i+1][1]-1   # Arithmetic returns
    logret[i] = np.log(df[i][1]/df[i+1][1])    # log-returns

# Arrays are not in the order i like; I prefer having most recent returns as last entries (bottom row)
ret = ret[::-1]
logret = logret[::-1]
year = year[::-1]

# Plotting returns:
returnsplot = plt.plot(year[2:6],ret)
plt.plot(year[2:6],logret)
plt.show()

## Exercise 2

r = 0.05
r2 = 0.10
beta = 1 / (1 + r)
beta2 = 1 / (1 + r2)
sigma = 0.15
sigma2 = 0.30
mu = 1
T = 60

@njit
def time_path(T):
    w = np.random.randn(T+1)  # w_0, w_1, ..., w_T
    w[0] = 0
    b = np.zeros(T+1)
    b2 = np.zeros(T+1)
    for t in range(1, T+1):
        b[t] = w[1:t].sum()
        b2[t] = w[1:t].sum()
    b = -sigma * b
    b2 = -sigma2 * b2
    c = mu + (1 - beta) * (sigma * w - b)
    c2 = mu + (1 - beta2) * (sigma2 * w - b2)
    return w, b, b2, c, c2

w, b, b2, c, c2 = time_path(T)

# Plotting everything in one chart, where dotted lines represent the time_path using new parameters.
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(mu + sigma * w, 'g-', label="Non-financial income")
ax.plot(c, 'k-', label="Consumption")
ax.plot( b, 'b-', label="Debt")
ax.plot(mu + sigma2 * w, 'g--', label="Non-financial income 2")
ax.plot(c2, 'k--', label="Consumption 2")
ax.plot( b2, 'b--', label="Debt 2")
ax.legend(ncol=3, mode='expand', bbox_to_anchor=(0., 1.02, 1., .102))
ax.grid()
ax.set_xlabel('Time')

plt.show()


# Simulation paths; I didn't manage to seperate between the old and new parameters
fig, ax = plt.subplots(figsize=(10, 6))

b_sum = np.zeros(T+1)
for i in range(250):
    w, b, b2, c, c2 = time_path(T)  # Generate new time path
    rcolor = random.choice(('c', 'g', 'b', 'k'))
    ax.plot(c, color=rcolor, lw=0.8, alpha=0.7)
    
ax.grid()
ax.set(xlabel='Time', ylabel='Consumption')

plt.show()



