import sim_homodyne as shd
import numpy as np
from matplotlib import pyplot as plt
from qutip import fock

# Dummy test program. 

# CURRENT VERSION: tests numerical inverse sample algorithm
# Generates N homodyne samples and plots reconstructed quadratures

N = 12
state = (fock(N,0) + fock(N,1))/np.sqrt(2)
theta = np.pi/4

# Define the quantum state
cd,xvec = shd.cumulative(state,theta,Npts = 1500)
cd0,xvec = shd.cumulative(fock(N,0),theta,Npts = 1500)

# Define the temporal mode and construct complete orthogonal basis
t = np.linspace(-1,1,100)
y = np.exp(-0.5*(t/.1)**2)
q = shd.generate_basis(y)


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

Nsmp = 2500

for k in range(5):
    photocurr = shd.gen_photocurrent(q,cd,cd0,xvec)
    ax1.plot(t,photocurr)
ax1.plot(t,y,lw=3,color='red',linestyle='--')

quads = np.zeros(Nsmp)

for k in range(Nsmp):
    photocurr = shd.gen_photocurrent(q,cd,cd0,xvec)
    quads[k] = np.dot(y,photocurr)

ax2.hist(quads,30,orientation='horizontal')