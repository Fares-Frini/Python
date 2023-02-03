import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
x=sp.symbols('x')
I=sp.integrate(2*x/(1+x**2),(x,1,2)).evalf()
print(I)
print(sp.integrate(2*x/(1+x**2),(x,1,2)))
A=np.array([[1,1,1],[1,3/2,2],[1,9/4,4]])
print(np.linalg.det(A))
b=np.array([[1],[3/2],[7/3]])
print(np.linalg.solve(A,b))
f=lambda a : 2*x/(1+x**2)
Q=2*1/(1+1**2)*0.16666667+2*(3/2)/(1+(3/2)**2)*0.66666667+2*2/(1+2**2)*0.16666667
print(np.abs(I-Q))
f=lambda xk : 2*xk/(1+xk**2)
a=1
b=2
n=5

def gauchecomposite(f,a,b,n):
    h=(b-a)/n
    R=0
    for i in range(n):
        xk=a+i*h
        R+=f(xk)
    return(h*R) 
GC=gauchecomposite(f,1,2,5)
print(GC)
print(abs(I-GC))
X=np.linspace(1,2,100)
g=lambda x: abs((2-2*x**2)/(x**2+1)**2)
plt.plot(X,g(X),'-r',linewidth=3,markersize=12)
plt.grid(True)
plt.xlabel('axe des abscisses ')
plt.ylabel('axe des ordonnées')
plt.title('Courbe de g')
plt.show()