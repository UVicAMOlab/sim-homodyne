# Simulates real time acquisition of homodyne data using a given temporal mode.

import sim_homodyne as shd
import numpy as np
from matplotlib import pyplot as plt, animation

# Define modes here
t = np.linspace(-1,1,100)
y = np.exp(-0.5*(t/.1)**2)
q = shd.generate_basis(y)
quads = np.array([])

# Define the marginal distributions
def marginal(*args):
    return shd.marginal_fock(1,float(args[0]))
def marginal_vac(*args):
    return shd.marginal_fock(0,float(args[0]))

# Set up plot canvas
fig, (ax1,ax2)= plt.subplots(1, 2, figsize=(10, 5))
# Plots
ln1, = ax1.plot([], [], lw=2)
ln1b, = ax1.plot(t, y, lw=2)
ln2, = ax2.plot([],[],'.',markersize=3)

# GTL for Animation
def init_animation():        
    #graph parameters for Ellipse
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-3.0, 3.0)
    ax2.set_ylim(-10.0, 10.0)
    return ln1,ln2,

def animate_fun(idx):
	global quads
	ih0 = shd.gen_photocurrent(q,marginal,marginal_vac)
	quads = np.append(quads,np.dot(ih0,y))
	ln1.set_data(t,ih0)
	ln2.set_data(1+np.arange(len(quads)),quads)
	ax2.set_xlim(0, max(10,len(quads)))
	return ln1,ln2,
    
annie = animation.FuncAnimation(fig,animate_fun,init_func = init_animation, interval = 10)
plt.show()