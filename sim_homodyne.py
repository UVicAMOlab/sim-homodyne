def cumulative(state,theta,spn=10,Npts = 2500):
    import numpy as np
    from scipy import integrate
    from qutip import wigner
    from scipy.ndimage import rotate

    xvec = np.linspace(-1,1,Npts)*spn
    W = wigner(state,xvec,xvec)
    Wr = rotate(W,180*theta/np.pi)
    Lr = Wr.shape[0]
    L0 = W.shape[0]
    chp = round((Lr-L0)/2)
    Wr = Wr[chp:chp+L0,chp:chp+L0] 
    large_marge = integrate.trapz(Wr,xvec,axis=0)
    return integrate.cumtrapz(large_marge,xvec,initial=0), xvec



def invtrans_sample(cdf,xvals):
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

