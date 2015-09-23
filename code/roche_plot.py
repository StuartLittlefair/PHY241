import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from trm import roche
from trm.subs import Vec3

plt.style.use('bmh')

x = np.linspace(-1,2,1000)
y = np.linspace(-1,1,1000)
X,Y = np.meshgrid(x,y)

def f(x,y):
    return roche.rpot(0.3,Vec3(x,y,0))
fv = np.vectorize(f)
Z = fv(X,Y)

levels = np.linspace(0.0,-2.5,50)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X,Y,Z,50,cmap='binary',levels=levels)
ax.set_zlim((-2.5,-0.5))
plt.show()
