
from qutip import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from scipy.integrate import trapz,cumtrapz

N = 12
rho = (fock(N,0) + fock(N,1))/np.sqrt(2)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

xvec = np.linspace(-5,5,200)
W = wigner(rho, xvec, xvec)


Wr = rotate(W,45)
Lr = Wr.shape[0]
L0 = W.shape[0]
chp = round((Lr-L0)/2)
Wr = Wr[chp:chp+L0,chp:chp+L0]

marge = trapz(Wr,xvec,axis=0)

cumlt = cumtrapz(marge,xvec,initial=0)

print(f'shape of original Wigner fn is {np.shape(W)}')
print(f'shape of rotated Wigner fn is {np.shape(Wr)}')

ax1.contourf(Wr,100)
ax2.plot(xvec,marge)
ax2.plot(xvec,cumlt)

plt.show()