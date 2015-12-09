# find the minimum of the rosenbrock function
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from lmfit import minimize, Parameters, Parameter, report_fit

# Chose a model that will create bimodality.
def func(pars,x,y,e):
    a = pars['a'].value
    b = pars['b'].value
    # Term b*b will create bimodality.
    model = a + b*b*x
    return (y-model)
    
def chi2(a,b,x,y,e):
    model = a + b*b*x
    res = (y-model)
    return np.sum(res**2)

# Create toy data for curve_fit.
x = np.array([0.0,1.0,2.0,3.0,4.0,5.0])
y = np.array([0.1,0.9,2.2,2.8,3.9,5.1])
e = np.array([1.0,1.0,1.0,1.0,1.0,1.0])

amin = -4.0  # minimal value of a covered by grid
amax = +4.0  # maximal value of a covered by grid
bmin = -3.0  # minimal value of b covered by grid
bmax = +3.0  # maximal value of b covered by grid

steps = 101
chi2_grid = np.zeros([steps,steps])
a = np.linspace(amin,amax,steps)
b = np.linspace(bmin,bmax,steps)
for i in range(steps):
    for j in range(steps):
        chi2_grid[steps-1-j,i] = chi2(a[i],b[j],x,y,e)


X, Y = np.meshgrid(a, b)
Z = chi2_grid
 
 


avals=[]
bvals=[]
cvals=[]
def iter_cb(pars,iter,resid, *args, **kws):
    global avals, bvals
    a = pars['a'].value
    b = pars['b'].value
    avals.append(a)
    bvals.append(b)
    cvals.append(chi2(a,b,x,y,e))

pars = Parameters()
pars.add('a',value=-2.4)
pars.add('b',value=-0.2) 
result = minimize(func, pars, args=(x, y, e),iter_cb=iter_cb,factor=0.01,ftol=1.0e-25)

#for av,bv,cv in zip(avals,bvals,cvals):
#    print(av,bv,cv)
av1 = avals
bv1 = bvals

avals = []
bvals = []
pars = Parameters()
pars.add('a',value=-2.4)
pars.add('b',value=0.4)
result = minimize(func, pars, args=(x, y, e),iter_cb=iter_cb,factor=0.01,ftol=1.0e-25)
av2 = avals
bv2 = bvals

fig = plt.figure()
threed=False
if threed:
    ax = plt.axes(projection='3d')
    ax.contour3D(X,Y,np.log10(Z),100,cmap='binary')
    ax.scatter3D(avals,bvals,np.log10(cvals))
    ax.plot3D(avals,bvals,np.log10(cvals))
else:
    ax = plt.axes()
    mpl.rcParams['contour.negative_linestyle'] = 'solid'
    ax.contour(X,Y,np.log10(Z),20,colors='b',alpha=0.2)
    ax.imshow(np.log10(Z),extent=[amin, amax, bmin, bmax],cmap='gray')
    ax.scatter(av1,bv1,color='r')
    ax.plot(av1,bv1,color='r')
    
    ax.scatter(av2,bv2,color='r',marker='+')
    ax.plot(av2,bv2,'r:')
    ax.set_ylim(-2,2)
    ax.set_xlim(-3,3)
    ax.set_xlabel('a')
    ax.set_ylabel('b')

plt.show()
