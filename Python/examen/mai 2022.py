import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numpy.polynomial import Polynomial


def baseNewton(X):


    w0 = Polynomial([1])
    w1 = Polynomial([-X[0], 1])
    w2 = Polynomial([X[0]*X[1], -(X[0]+X[1]), 1])
# pour w2 on peut utiliser la multiplication de deux polynômes
    w = [w0, w1, w2]
    return w


def polynomeNewton(X, Beta):


    p = Polynomial([0])
    N = baseNewton(X)
    n=2
    for i in  range(3):
        p+=(N[i]*Beta[i])
    return p

Beta =[0.2,0.3,0.1]
X=[-2,-1,0]
A=polynomeNewton(X,Beta)
print(A)
f=lambda x : 1/(1+x**2)
print(abs(f(1/2)-A(1/2)))
Y=[f(-1),f(-2),f(0)]
# Y est la liste qui contient f(-2), f(-1) et f(0)
t=np.linspace(-2,1,100)
plt.plot(t,f(t),'-r',t,A(t),'-b',-2, f(-2),'*g', -1, f(-1),'*g' ,0, f(0),'*g')
plt.xlabel('axe des abscisses')
plt.ylabel('axe des ordonnées')
plt.legend(('f','P','points d''interpolation'))
plt.title('Interpolation de Newton')
plt.show()
def formulecomposite(f,df,a,b,n):
    h=(b-a)/n
    A=0
    B=0
    for i in range(1,n) :
        A+=f(a+i*h)
    for i in range(0,n) :
        B+=f(a+(i+0.5)*h)
    fp=sp.diff(f)
    return (h/60)*(14*(f(a)+f(b))+h(fp(a)-fp(b))+28*A+32*B)

w=sp.symbols('w')
I=sp.integrate(sp.cos(2*w)*sp.exp(-w),(w,-1,0)).evalf()
print(I)
x=sp.symbols('x')
f=lambda x:sp.cos(2*x)*sp.exp(-x)
#ou f= sp.Lambda(x, sp.cos(2*x)*sp.exp(-x) )
a=-1
b=0
I=sp.integrate(f(x) ,(x,a,b) ).evalf()
print(I)
fp=sp.diff(f(x),x)
a=-1
print(fp)
df=sp.Lambda(x,sp.diff(f(x),x)) # expression symbolique de f'
a=-1
b=0
print(df)
print(df(a).evalf())
print(df(b).evalf())