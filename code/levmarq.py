# find the minimum of the rosenbrock function
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from lmfit import minimize, Parameters, Parameter, report_fit

# Chose a 1D model with bimodality
def func(pars,x,y,e):
    a = pars['a'].value
    # Term a*a will create bimodality.
    model = a*a*x
    return (y-model)
    
def chi2(a,x,y,e):
    model = a*a*x
    res = (y-model)
    return np.sum(res**2)

# Create toy data for curve_fit (a = 2)
x = np.array([0.0,1.0,2.0,3.0,4.0,5.0])
y = 4*x + np.random.normal(size=len(x))
e = np.array([1.0,1.0,1.0,1.0,1.0,1.0])

amin = -4.0  # minimal value of a covered by grid
amax = +4.0  # maximal value of a covered by grid

steps = 1001
a = np.linspace(amin,amax,steps)
chi2_grid = np.array([chi2(av,x,y,e) for av in a])

avals=[]
cvals=[]
def iter_cb(pars,iter,resid, *args, **kws):
    global avals
    a = pars['a'].value
    avals.append(a)
    cvals.append(chi2(a,x,y,e))

pars = Parameters()
pars.add('a',value=0.3)
result = minimize(func, pars, args=(x, y, e),iter_cb=iter_cb,factor=0.01,ftol=1.0e-25)

for av,cv in zip(avals,cvals):
    print(av,cv)
av1 = avals
cv1 = cvals

avals = []
cvals = []
pars = Parameters()
pars.add('a',value=-0.3)
result = minimize(func, pars, args=(x, y, e),iter_cb=iter_cb,factor=0.01,ftol=1.0e-25)
av2 = avals

fig = plt.figure()
ax = plt.axes()
ax.plot(a,np.log10(chi2_grid),'k-')
ax.scatter(av1,np.log10(cv1),color='r')
ax.plot(av1,np.log10(cv1),'r-')
ax.set_xlabel('Parameter value')
ax.set_ylabel('Chisq')

plt.show()
