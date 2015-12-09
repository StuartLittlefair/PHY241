import numpy as np
from scipy import integrate

def transit(t,Rp,Rs,i,P,t0,mu):
    """Calculate the transit shape due to an exoplanet.
    
    This function implements the model of exoplanetary transits found in
    Sackett et al (1999, ASIC, 532, 189). 
    
    Args
    ----
    t: np.ndarray, float
        the times at which to calcualte the transit lightcurve. Units of days (MJD)
    Rp: float
        radius of exoplanet, divided by the orbital separation
    Rs: float
        radius of star, divided by the orbital separation
    i: float
        inclination, in degrees
    P: float
        Orbital period of exoplanet, in days
    t0: float
        Time of mid-transit. Units of days (MJD)
    mu: float
        linear limb-darkening parameter for star
    
    Returns
    --------
    np.ndarray, float
        the transit lightcurve, normalised to 1 outside of transit
    """
    t = np.atleast_1d(t)
    return np.array([_transit(tval,Rp,Rs,i,P,t0,mu) for tval in t])
    
def _transit(t,Rp,Rs,i,P,t0,mu):
    """Calculates transit at single time"""
    t -= t0
    i = np.radians(i)
    om = 2*np.pi/P
    d = np.sqrt( np.sin(om*t)**2 + np.cos(i)**2 * np.cos(om*t)**2 )

    if d>Rs+Rp:
        return 1.0
        
    def LD(x,mu):
        return 1-mu*(1-np.sqrt(1-(x/Rs)**2))
        
    def integrand(x):
        costheta = (d**2 + x**2 - Rp**2) / 2. / x / d
        if  costheta < -1:
            theta = np.pi
        else:
            theta = np.arccos(costheta)
        return x*LD(x,mu)*theta
    
    # bottom integral in eqn 12 of sackett reduces to
    fac = np.pi*(3-mu)*Rs**2/6.
    
    lolim = max(0,d-Rp)
    uplim = min(Rs,d+Rp)
    
    A, err =  integrate.quad(integrand,lolim,uplim)
    return 1-A/fac
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-muted')
    
    from astropy import units as u, constants as const
    # WASP-4b
    a = 0.02312*u.au
    rp = 0.02809245
    Rs = 0.18271515
    P = 1.38823187
    i = 88.6
    mu=0.311
    t0 = 54748.150490
    t = np.linspace(t0-4./24.,t0+4./24,1000)
    y = transit(t,rp,Rs,i,P,t0,mu)
    x,yd,e = np.loadtxt('wasp4_transit.txt').T
    plt.errorbar(x,yd,yerr=e,fmt='.')
    plt.plot(t,y)
    plt.show()