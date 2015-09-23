# see http://spacemath.gsfc.nasa.gov/Calculus/10Page121.pdf
import numpy as np

# spacecraft angles (one orbit)
theta = np.linspace(0.0,2.0*np.pi,20)

# times
t = 9*(np.degrees(theta) - 0.55*np.sin(theta))/2./np.pi

# distance
r = 5.7 - (210 / (100-55*np.cos(theta)))

# radiation
g = 0.136*r**6 - 2.194*r**5 + 13.89*r**4 - 43.73*r**3 + 71.78*r**2 - 57.95*r + 18.15

np.savetxt('radiation_dose.txt', np.column_stack((t,g)))
