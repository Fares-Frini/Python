import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numpy.polynomial import Polynomial
def somme(x,y):
    xi=0
    yi=0
    n=len(x)
    xy=0
    x2=0
    for i in range(n):
        print(i)
        xi+=x[i]
        yi+=y[i]
        xy+=x[i]*y[i]
        x2+=x[i]**2
    xbar=xi/n
    ybar=yi/n
    xybar=xy/n
    x2bar=x2/n

    return(xbar,ybar,xybar,x2bar)
def coefificient(xbar,ybar,xybar,x2bar):
    b=(xybar-xbar*ybar)/(x2bar-(xbar**2))
    a=ybar-b*xbar
    return a,b
        
x=[1,2,3,4,5,6,7,8,9,10,11,12]
y=[4.3,5.1,5.7,6.3,6.8,7.1,7.2,7.2,7.2,7.2,7.5,7.8]
xbar,ybar,xybar,x2bar=somme(x,y)

print(xbar)
print(ybar)
print(xybar)
print(x2bar)
a=coefificient(xbar,ybar,xybar,x2bar)[0]
b=coefificient(xbar,ybar,xybar,x2bar)[1]
print(a)
print(b)
def schema(f,x0,t0,T,N):
    X=[x0]
    h=T/N
    tk=[t0]
    p=[x0+h*f(t0,x0)]
    for i in range(N-1):
        tk.append(t0+(i+1)*h)
        X.append(X[i]+(h/2)*(f(tk[i],x[i])+f(tk[i+1],p)))
        
        p.append(X[i+1]+h*f(tk[i+1],X[i+1]))
    return(X)

r=2
K=100
f=lambda t, w: (w)*(1-w/K)
X=schema(f,40,0,72,180)
        
        