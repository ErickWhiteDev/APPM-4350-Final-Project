import matplotlib.pyplot as plt
from scipy.special import eval_laguerre
import numpy as np

x = np.linspace(-5,20,500)

fig = plt.figure()
ax = fig.subplots(1,1)

for n in range(5):
    plt.plot(x,eval_laguerre(n,x),label=f"$L_{n}$")

ax.set_ylim(-10,20)
ax.grid()

plt.xlabel('$x$')
plt.ylabel("$L_n(x)$")
plt.legend()
plt.title('Standard Laguerre Polynomials')

fig = plt.figure()
ax = fig.subplots(1,1)

for n in range(5):
    plt.plot(x,np.exp(-x/2)*x**((n+1)/2)*eval_laguerre(n,x),label=f"$L_{n}$")

ax.set_ylim(-10,20)
ax.grid()

plt.xlabel('$x$')
plt.ylabel("$L_n(x)$")
plt.legend()
plt.title('Modified Laguerre Polynomials')
plt.show()
