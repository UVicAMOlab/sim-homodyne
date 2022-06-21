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

def inv_smp_num(cdf,xvals):
    '''
    Draws a sample from a probability density fn with cumulative cdf
    Numerical implementaion of Inverse Sample Transform technique
    
        Parameters:
            cdf (array): array representing the cumulative distribution of P(x)
            xvals (array): x values corresponding to cdf(xvals)
        
        Returns:
            (float): random number weighter accoring to P(x)
    '''
    # 1. Generate uniform random number on [0,1]
    import numpy as np
    u = np.random.rand() 
    # Most probable guess is at mean of CDF, Start search here and search from [xL,xR] = entire array
    x0 = int(np.argwhere(cdf>=np.mean(cdf))[0,0]) # start at mean of distribution
    xL = 0
    xR = len(cdf)-1
    
    # Perform binary search via bisection since cdf is monatomic
    while (xR-xL) > 1:
    # If guess is too high, value is in left half. Chop bounds and recenter
        if(cdf[x0] > u):
            xR = x0 
            x0 = int(np.round((x0+xL)/2))
    # Otherwie on right side
        else:
            xL = x0
            x0 = int(np.round((x0+xR)/2))
    return xvals[x0]
    
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

