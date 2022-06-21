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
cd,xvec = shd.cumulative(state,theta)
# cd0,xvec = shd.cumulative(fock(N,0),theta)

# Simulate Nsmp homodyne samples
Nsmp = int(1e4)
y = np.zeros(Nsmp)


for k in range(Nsmp):
    y[k] = shd.invtrans_sample(cd,xvec)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

ax1.plot(y,'.',markersize='2')
ax2.hist(y,30,orientation='horizontal')

# plt.show()

# Define the marginal distributions
# def marginal(*args):
#     return shd.marginal_fock(1,float(args[0]))
# def marginal_vac(*args):
#     return shd.marginal_fock(0,float(args[0]))


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