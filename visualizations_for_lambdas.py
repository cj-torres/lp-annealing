import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from numpy import meshgrid

delta = 0.0025
lrange = arange(0.0, 2.0, delta)
xrange = arange(0.0, .75, delta)
L, X = meshgrid(lrange, xrange)

lam = np.logspace(np.log10(.495), -6, 100)

F = X
G = L * X ** (L-1)

cmap = plt.get_cmap('viridis')
log_lam = np.log(lam)
norm = plt.Normalize(vmin=min(log_lam), vmax=max(log_lam))

plt.figure(figsize=(10, 8))

for lam_val in lam:
    color = cmap(norm(np.log(lam_val)))

    # Plot the contour for each lambda value
    plt.contour(L, X, (F-lam_val*G), levels=[0], colors=[color], label=f'λ={lam_val:.5f}')

L_curve = np.linspace(0.01, 2.0, 1000)  # Avoid division by zero
X_curve = np.exp(-1 / L_curve)
plt.plot(L_curve, X_curve, color='red', linestyle='--', linewidth=2, label='Maxima')


# Add labels and title
plt.xlabel('Lp (semi)norm')
plt.ylabel('Sparsifying neighborhood size near zero')
plt.title('Lp (semi)norms and their sparsifying regions parameterized by λ')
# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='log(λ)')

plt.legend(loc='best')
plt.grid(True)
plt.show()

plt.close()


delta = 0.00025
lrange = arange(0.0, .25, delta)
xrange = arange(0.0, .1, delta)
L, X = meshgrid(lrange, xrange)

#lam = np.logspace(-.305, -6, 75)

F = X
G = L * X ** (L-1)

cmap = plt.get_cmap('viridis')
log_lam = np.log(lam)
norm = plt.Normalize(vmin=min(log_lam), vmax=max(log_lam))

plt.figure(figsize=(10, 8))

for lam_val in lam:
    color = cmap(norm(np.log(lam_val)))

    # Plot the contour for each lambda value
    plt.contour(L, X, (F-lam_val*G), levels=[0], colors=[color], label=f'λ={lam_val:.5f}')

L_curve = np.linspace(0.01, .25, 1000)  # Avoid division by zero
X_curve = np.exp(-1 / L_curve)
plt.plot(L_curve, X_curve, color='red', linestyle='--', linewidth=2, label='Maxima')


# Add labels and title
plt.xlabel('Lp (semi)norm')
plt.ylabel('Sparsifying neighborhood size near zero')
plt.title('Lp (semi)norms and their sparsifying regions parameterized by λ')
# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='log(λ)')

plt.legend(loc='best')
plt.grid(True)
plt.show()

plt.close()


L_curve = np.linspace(0.01, 2.0, 1000)
lam_curve = 1/(L_curve*np.exp(2/L_curve - 1))
x_curve = np.exp(-1/L_curve)
log_x = np.log(x_curve)

plt.figure(figsize=(10, 8))
plt.scatter(L_curve, lam_curve, color=cmap(x_curve))
norm = plt.Normalize(vmin=min(log_x), vmax=max(log_x))

# Add labels and title
plt.xlabel('Lp (semi)norm')
plt.ylabel('λ for which Lp (semi)norm has the maximal sparsifying neighborhood')
plt.title('Lp (semi)norms and their sparsifying regions parameterized by λ')
# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='log(w)')

plt.show()

