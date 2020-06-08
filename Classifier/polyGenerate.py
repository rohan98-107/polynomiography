import numpy as np
import matplotlib.pyplot as plt
import random, string
from PIL import Image

def ddx(f,xnot):
	h = 0.00000000001
	return (f(xnot+h)-f(xnot))/h

def Newton(z,f,max_iter=100,eps=1e-5):
	for i in range(max_iter):
		step = f(z)/ddx(f,z)
		if abs(step) < eps:
			return i, z
		z -= step
	return i, z

def plot_iters(p, pprime, n=200, extent=[-1,1,-1,1], cmap='gray'):
    """Shows how long it takes to converge to a root using the Newton-Rahphson method."""
    m = np.zeros((n,n))
    xmin, xmax, ymin, ymax = extent
    for r, x in enumerate(np.linspace(xmin, xmax, n)):
        for s, y in enumerate(np.linspace(ymin, ymax, n)):
            z = x + y*1j
            m[r, s] = newton(z, p, pprime)[0]
    plt.imshow(m.T, cmap=cmap, extent=extent)
    plt.show()

def plot(p, n=700, extent=[-1.5,1.5,-1.5,1.5], cmap='Reds',save=False):
    """Shows basin of attraction for convergence to each root using the Newton-Raphson method."""
    root_count = 0
    roots = {}

    m = np.zeros((n,n))
    xmin, xmax, ymin, ymax = extent
    for r, x in enumerate(np.linspace(xmin, xmax, n)):
        for s, y in enumerate(np.linspace(ymin, ymax, n)):
            z = x + y*1j
            shade, root = Newton(z,p)
            root = np.round(root, 3)
            if not root in roots:
                roots[root] = root_count
                root_count += 1
            m[r, s] = roots[root] + shade
    if not save:
        plt.imshow(m.T, cmap=cmap, extent=extent)
        plt.show()
    else:
        s = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        plt.imsave( "images/"+ s + ".png",m.T,cmap=cmap)
        return s + ".png"

# DEMO GRAPHS --------------------------------------------------

#flower
#f = lambda z: z**5 - 1
#plot(f,cmap='twilight',extent=[-.75,.75,-.75,.75])

#butterfly
#g = lambda z: z**9 + z**2 - 4*z + 5
#plot(g,cmap='inferno',extent=[-1.1,-0.8,-0.2,0.2])

#building/tower
#k = lambda z: (z-5)**3 + z - 1
#plot(k,cmap='brg',extent=[7,30,-5,5])

#dog
#h = lambda z: (z)**4 + z - 1
#plot(h,cmap='tab20c',extent=[-0.8,1.5,-0.9,0.9])

#----------------------------------------------------------------

#misc graphs - for demo -
#f = lambda z: z**3 - 1
#plot(h)
