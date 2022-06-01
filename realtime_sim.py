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

fig = plt.figure()
spc = fig.add_gridspec(ncols =2,nrows = 2)
ax_trc = fig.add_subplot(spc[0,:])
ax_pts = fig.add_subplot(spc[1,0])
ax_hist = fig.add_subplot(spc[1,1])

# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# Plots
ln1, = ax_trc.plot([], [], lw=2)
ln1b, = ax_trc.plot(t, y, lw=2)
ln2, = ax_pts.plot([],[],'.',markersize=3)
h1 = ax_hist.hist(np.ones(20))

# GTL for Animation
def init_animation():        
    ax_trc.set_xlim(-1.0, 1.0)
    ax_trc.set_ylim(-3.0, 3.0)
    ax_pts.set_ylim(-10.0, 10.0)
    return ln1,ln1b,ln2,h1

def animate_fun(h_bars):
	def animate(idx):
		global quads
		ih0 = shd.gen_photocurrent(q,marginal,marginal_vac)
		quads = np.append(quads,np.dot(ih0,y))
		ln1.set_data(t,ih0)
		ln2.set_data(1+np.arange(len(quads)),quads)
		ax_pts.set_xlim(0, max(10,len(quads)))
		return ln1,ln2,
	return animate()
    
annie = animation.FuncAnimation(fig,animate_fun(barz),init_func = init_animation, interval = 10)
plt.show()