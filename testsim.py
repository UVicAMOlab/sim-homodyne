import sim_homodyne as shd
import numpy as np
from matplotlib import pyplot as plt
import time

N = 10000
def marginal(*args):
    return shd.marginal_fock(1,float(args[0]))
def marginal_vac(*args):
    return shd.marginal_fock(0,float(args[0]))
quadz = np.zeros(N)
for k in range(N):
    quadz[k] = shd.inv_trans_smp(marginal)


t = np.linspace(-1,1,100)
y = np.exp(-0.5*(t/.1)**2)


q = shd.generate_basis(y)

# def marginal():
#     eta = .75
#     return eta*shd.marginal_fock(1) + (1-eta)*shd.marginal_fock(0)

ih = np.zeros(len(t))
Niter = 100000

quads = np.zeros(Niter)

start_time = time.time()
print(f'Starting loop of {Niter} iterations')
for k in range(Niter):
    ih0 = shd.gen_photocurrent(q,marginal,marginal_vac)
    quads[k] = np.dot(ih0,y)
    ih += ih0**2
stop_time = time.time()

dlt = stop_time-start_time

print(f'Finished in {round(dlt,2)} seconds')
print(f'{round(1e3*dlt/Niter,2)} ms/loop')

fig,(ax1,ax2) = plt.subplots(1,2,figsize=[9,4])

ax1.plot(t,ih/Niter)
ax1.plot(t,(y/np.sqrt(Niter))+1/2)
ax2.plot(quads,'.',markersize=2)
# for k in range(45,55):
#     plt.plot(t,q[:,k])
# plt.plot(t,q[-1])
plt.show()