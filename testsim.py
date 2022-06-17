import sim_homodyne as shd
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import integrate

# Dummy test program. 

# CURRENT VERSION: tests numerical inverse sample algorithm
# Generates N homodyne samples and plots reconstructed quadratures

spn = 10
Npts = 10000
x = np.linspace(-1,1,Npts)*spn
pr = 2*x**2 * np.exp(-x**2)/np.sqrt(np.pi)
assert abs(integrate.trapz(pr,x)-1) < 1e-4,"\nProbability distribution appears to be unnormalized.\n Try extending range"

cd = integrate.cumtrapz(pr,x)
cd = np.append(cd,cd[-1])



fig,(ax1,ax2) = plt.subplots(1,2,figsize=[10,4])

ax1.plot(x,cd)
ax1.plot(x,pr)

Nsmp = int(1e3)
y = np.zeros(Nsmp)

for k in range(Nsmp):
    y[k] = shd.inv_smp_num(cd,x)

ax2.hist(y,200)

plt.show()

# Define the marginal distributions
# def marginal(*args):
#     return shd.marginal_fock(1,float(args[0]))
# def marginal_vac(*args):
#     return shd.marginal_fock(0,float(args[0]))

# # Define the temporal mode and define complete orthogonal basis
# t = np.linspace(-1,1,100)
# y = np.exp(-0.5*(t/.1)**2)
# q = shd.generate_basis(y)

# ih_var = np.zeros(len(t))

# # Generate Niter samples
# Niter = 1000
# quads = np.zeros(Niter) # Photocurrent

# # Start performance measurement
# start_time = time.time()
# print(f'Starting loop of {Niter} iterations')

# # Main loop:  
# for k in range(Niter):
#     ih0 = shd.gen_photocurrent(q,marginal,marginal_vac) #generate a trace
#     quads[k] = np.dot(ih0,y) # Quadrature reconstructed by integrating along mode
#     ih_var += ih0**2  # variance
# stop_time = time.time()

# dlt = stop_time-start_time

# print(f'Finished in {round(dlt,2)} seconds')
# print(f'{round(1e3*dlt/Niter,2)} ms/loop')

# # Plot-it!
# fig,(ax1,ax2) = plt.subplots(1,2,figsize=[9,4])
# ax1.plot(t,ih_var/Niter)
# ax1.plot(t,(y*2/np.sqrt(Niter))+1/2)
# ax2.plot(quads,'.',markersize=2)
# plt.show()