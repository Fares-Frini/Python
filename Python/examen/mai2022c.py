import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numpy.polynomial import Polynomial

def baseNewton(X):
    w0=Polynomial([1])
    w1=Polynomial([-X[0],1])
    w2=Polynomial([X[0]*X[1],-(X[0]+X[1]),1])
# pour w2 on peut utiliser la multiplication de deux polynômes
    w=[w0,w1,w2]
    return w

def polynomeNewton(X,Beta):
    p=Polynomial([0])
    n=len(X)
    N=baseNewton(X)
    for i in np.arange(0,n):
        produit=Beta[i]*N[i]
        p+=produit
    return p

X=[-2,-1,0]
Beta=[0.2,0.3,0.1]
p=polynomeNewton(X,Beta)
print(p)

f= lambda x: 1/(1+x**2)
E=abs(p(0.5)-f(0.5))
E
Y=[f(-2),f(-1),f(0)]
t=np.linspace(-2,1,100)
plt.plot(t,f(t),'b',t,p(t),'r',X,Y,'*')
plt.xlabel('axe des abscisses')
plt.ylabel('axe des ordonnées')
plt.legend(('f','P','points d''interpolation'))
plt.title('Interpolation de Newton')

def formulecomposite(f,df,a,b,n):
    h=(b-a)/n
    Q= 14*(f(a)+f(b)) + h*(df(a)-df(b))
    for k in range(1,n):
        Q+= 28*f(a+k*h)
    for k in range(0,n):
        Q+= 32*f(a+(k+0.5)*h)
    return (h/60)*Q

x=sp.symbols('x')
f=lambda x:sp.cos(2*x)*sp.exp(-x)
#ou f= sp.Lambda(x, sp.cos(2*x)*sp.exp(-x) )
a=-1
b=0
I=sp.integrate(f(x) ,(x,a,b) ).evalf()
print(I)
df=sp.Lambda(x,sp.diff(f(x),x)) # expression symbolique de f'
a=-1
b=0
print(df)
print(df(a).evalf())
print(df(b).evalf())
# Donner les instructions nécessaires
EQ=np.abs(formulecomposite(f,df,a,b,30).evalf()-I)
print('E_Q(f)=',EQ)
def nombreintervalles(f,df,a,b,epsilon):
    n=1
    E=np.abs(formulecomposite(f,df,a,b,n).evalf()-I)
    while E > epsilon:
        n+=1
        E=np.abs(formulecomposite(f,df,a,b,n).evalf()-I)
    return n
NQ=[]
epsilon = (1/10)**np.arange(3,10,2)
for eps in epsilon:
NQ.append(nombreintervalles(f,df,a,b,eps))
plt.grid(True)
plt.plot(epsilon,NQ,'ro-')
plt.xlabel('epsilon ')
plt.xscale('log')
plt.ylabel('Nombre de sous intervalles')
plt.title('Nombre de sous intervalles en fonction de epsilon')

def erreurquadrature(a,b,n,k):

    x=sp.symbols('x')
    P=lambda x: x**k
    dP=sp.Lambda(x,sp.diff(P(x),x))
    I1=sp.integrate( P(x) ,(x,a,b) ).evalf()
    I2=formulecomposite(P,dP,a,b,n).evalf()
    Erreur=np.abs(I1-I2)
    return Erreur

def degréprécision(a,b,n):

    k=0
    Erreur=erreurquadrature(a,b,n,k)
    while Erreur ==0 :
        k=k+1
        Erreur=erreurquadrature(a,b,n,k)
    return k-1 # c'est le degré de précision d