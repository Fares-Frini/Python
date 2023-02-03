import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

t=np.linspace(-1.5*np.pi,1.5*np.pi,1000)
plt.figure(figsize=(20,10))
plt.plot(t,np.sin(t),'-mo',linewidth=3,markersize=12)


plt.xlabel('t',fontsize=30)
plt.xticks(fontsize=10) 
plt.yticks(fontsize=10)
plt.grid(True)
plt.legend(('Polynôme d\'interpolation de Lagrange', 'cosinus', 'points d\'interpolation'),fontsize=20, loc = 0)
plt.show() # affiche la figure à l'écran

