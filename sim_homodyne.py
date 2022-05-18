def inv_trans_smp(marg,xL = -10, xR = 10,thrsh = 1e-4,x0 = 0):
    '''
    Performs a sample of a probability distribution P(x) given its marginal
        marg is a function representing the marginal distribution
        xL and xR are limits for bisection. Most distns here will be mostly within [-3,3]
        thrsh is the threshold for the bisection search. If within trsh, accept this value
        x0 is the starting point for bisection this should be the mean of the distribtion.
    '''
    from numpy import random as rnd 
    # These are the search limits

    u = rnd.rand()
    x = x0
    while abs(marg(x) - u) > thrsh: 
        if marg(x) < u:
            xL = x
            x = (x+xR)/2
        else:
            xR = x
            x = (x+xL)/2
    return x

def marginal_fock(n,x):
    ''' 
        Provides a marginal distribution of the n photon Fock state
        TODO: Find a way to generalize ... is there a recipe for generating?
    '''
    from scipy import special as spfn
    import numpy as np

    if x is None:
        x = np.linspace(-5/2,5/2,250)

    if n == 0:
        return (1+spfn.erf(x))/2
    elif n ==1:
        return (1+spfn.erf(x))/2 - x*np.exp(-x**2)/np.sqrt(np.pi)
    else:
        print('ERROR!\n\tn>1 not currently supported')
        return 0

def generate_basis(y):
    import numpy as np
    N = len(y)
    # Define initial detector basis modes as instants of time
    x = np.zeros([N,N])
    for k in range(1,N):
        x[k][k] = 1
    x[:,0] = y
    q,r = np.linalg.qr(x)
    return q

def gen_photocurrent(modes,marg,marg_vac,scl = 1):
    '''
        Simulate a single time domain trace from a homodyne detector.
        modes: a set of orthonormal modes from generate_basis()
        marg: the marginal distribution corresponding to the quantum state
        marg_vac: the background state. Normally this is the vacuum but could be thermal
        scl: can scale the homodyne photocurrent by this value
    '''
    import numpy as np
    Npts = len(modes)
    ihd = np.zeros(Npts)
    q0 = inv_trans_smp(marg)
    ihd += q0*modes[:,0]
    for k in range(1,Npts):
        q0 = inv_trans_smp(marg_vac)
        ihd +=q0*modes[:,k]
    return ihd*scl

