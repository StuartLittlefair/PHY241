import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

time_step = 0.005
period1 = 5
period2 = 3

time = np.arange(0,20,time_step)
sig = np.sin(2*np.pi*time/period1) +\
    0.2*np.sin(2*np.pi*time/period2) +\
    np.random.normal(scale=0.5,size=len(time))

np.savetxt('lightcurve.txt',np.column_stack((time,sig)))
plt.plot(time,sig)
plt.show()