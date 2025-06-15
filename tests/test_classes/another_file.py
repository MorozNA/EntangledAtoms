import numpy as np
from src.orbitals import PrimitiveGaussian, GaussianProduct, Orbital

sigmax = 1.0
sigmay = 1.0
alpha = np.array([1 / (2 * sigmax**2), 1 / (2 * sigmay**2)])
coeff = 1

rA = np.array([0.0, 0.0])

pgA = PrimitiveGaussian(alpha, coeff, rA)
pgA.coeff = pgA.coeff / np.sqrt((pgA * pgA).integrate())
varphiA1 = Orbital([GaussianProduct([pgA], [0])])
rho1 = varphiA1**2


def plot_rho1(x1, x2, rho):
    coordinates = np.array([x1, x2])
    return rho.get_value([coordinates])


plot_rho1_vec = np.vectorize(plot_rho1, excluded='rho')


from pylab import figure, cm
import matplotlib.pyplot as plt

x1_min = -2.1
x1_max = 2.1
x2_min = -2.1
x2_max = 2.1
step = 0.05

x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
print(x1.shape)
print(x2.shape)
y1 = plot_rho1_vec(x1, x2, rho1)

plt.imshow(y1, extent=[x1_min, x1_max, x2_min, x2_max], cmap=cm.jet, origin='lower', aspect='equal')
plt.colorbar()
plt.contour(x1, x2, y1, extent=[x1_min, x1_max, x2_min, x2_max], colors='black', origin='lower', linewidths=0.7)

plt.title(r'$\rho(x, y)$', fontsize=16)
plt.xlabel('x/a')
plt.ylabel('y/a')
plt.yticks(range(-2, 3, 1))

plt.scatter(rA[0], rA[1], s=40, c='black', marker='o', label=r'Atomic position')
#plt.savefig("evaluate_2d_function_using_meshgrid_03.png", bbox_inches='tight')
plt.legend(loc='lower left')
plt.show()
plt.show()

