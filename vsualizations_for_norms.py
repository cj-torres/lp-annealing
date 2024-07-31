import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from numpy import meshgrid

wrange = np.linspace(-2, 2, 100000)
norms = arange(.25, 2, .25)

plt.figure(figsize=(20, 8))

for norm in norms:
    l_p = np.abs(wrange) ** norm
    plt.plot(wrange, l_p, color='black', linestyle='--')

plt.grid(True, color='gray')
plt.show()

wrange = np.linspace(-2, 2, 100000)
norm = .5

l_p = np.abs(wrange) ** norm
plt.plot(wrange, l_p, color='black', linestyle='--')

plt.grid(True, color='gray')
plt.show()