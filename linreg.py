#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:33:17 2018

@author: richard.guinto@gmail.com
"""

import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#### Edit the ppar list to contain the polynomial coefficient
#### Assumption:  from 2 to 4 coefficents only (up to 3rd degree polynomial)
#### first value is the coefficient of the highest degree
#### for example:  [2, 4, -1] means y = 2x^2 + 4x - 1

#ppar = [87, -20, 30] #x2, x1, and x0 coef respectively

import sys

pcount = len(sys.argv)-1
if pcount > 4:
    pcount = 4
print("param count: ", pcount)
if pcount >= 2:
    ppar = list(range(pcount))
    for i in range(1,pcount+1):
        ppar[i-1] = int(sys.argv[i])
elif pcount == 1:
    print("Please enter more than 1 parameter")
    sys.exit()
else:
    #generate random parameters
    print("Parameters not found, generating random parameters...")
    ppar = np.random.randint(-30,30,3)
    
print("Polynomial parameters: ", ppar)
p = np.poly1d(ppar)



# h(x) = theta0 + theta1 x1 + theta2 x2 + theta3 x3
# where x1 = x, x2 = x^2, and x3 = x^3
r = range(-100,101)
x = np.ndarray(len(ppar))
X = np.ndarray([len(r), len(ppar)])
Y = np.ndarray([len(r)])
xi = 0
#this section of the code will generate the normalize dataset
# of the polynomial equation
# given the value of x in a specific range of integer values
for v in r:
    x[0] = 1
    for i in range(1,len(ppar)) :
        x[i] = v**i / np.abs(r[0]**i)
        y = p(v)
    X[xi,:] = x
    Y[xi] = y
    xi = xi + 1


plt.plot(r,Y)
#Add some uniform distribution noise in the output
Y = Y + np.random.uniform(-1,1,Y.shape)
plt.plot(r,Y)
#normalize output
Y = Y / np.abs(r[0]**len(ppar))
plt.show()
        
theta = np.random.randint(-10,10,size=len(ppar))
loss_rate = np.ndarray(len(theta))
temp = np.ndarray(len(theta))
predict = np.matmul(X,theta)
loss = predict - Y
max_iter = 10000
loss_total = np.ndarray(max_iter)
loss_total[0] = np.sum(loss**2)/(2* len(loss))
#alpha = (np.cos(0)/2+0.5)/1000
alpha = 0.3 / (10**(4 - len(ppar)))
for iter in range(1,max_iter):
    for i in range(0,len(theta)):
        loss_rate[i] = np.sum(loss * X[:,i])/len(loss)
        temp[i] = theta[i] - (alpha * (loss_rate[i]))

    theta = temp
    predict = np.matmul(X,theta)
    loss = predict - Y
    loss_total[iter] = np.sum(loss**2)/(2* len(loss))
    loss_diff = loss_total[iter] - loss_total[iter-1]
    #alpha = (np.cos(iter * np.pi / 50)/2+0.5)/1000
    #np.set_printoptions(precision=2,suppress=False)
    #print(theta, loss_total[iter])
    
print(theta, loss_total[max_iter-1])

#graph the loss value for the first hundreds of iterations
limit = int(1200/len(ppar))
iter = range(0,limit)
plt.plot(iter,loss_total[:limit])
plt.title("Loss Function")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

#mult = [10**8, 10**6, 10**4, 10**2]
mult = [10**2,10**4]
if len(ppar) >= 3:
    mult = mult + [10**6]
if len(ppar) >= 4:
    mult = mult + [10**8]
mult.reverse()
theta = theta * mult
print("theta: ", theta)

coef = theta[::-1]
coef = np.round(coef,0)
print("Polynomial Coefficients: ", coef)
