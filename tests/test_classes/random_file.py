import sympy as smp

g_i = smp.symbols('g_i')
g_j = smp.symbols('g_j')
g_k = smp.symbols('g_k')
g_l = smp.symbols('g_4')

e_i = smp.symbols('e_i')
e_j = smp.symbols('e_j')
e_k = smp.symbols('e_k')
e_l = smp.symbols('e_4')
Phi_0 = (g_i * e_k - g_k * e_i) * (g_j * e_l - g_l * e_j) + (g_j * e_k - g_k * e_j) * (g_i * e_l - g_l * e_i) + \
        (g_i * e_l - g_l * e_i) * (g_j * e_k - g_k * e_j) + (g_j * e_l - g_l * e_j) * (g_i * e_k - g_k * e_i)

Phi_0 = Phi_0 / 2

Phi_1 = (g_i * e_k + g_k * e_i) * (g_j * e_l + g_l * e_j) - (g_j * e_k + g_k * e_j) * (g_i * e_l + g_l * e_i) - \
        (g_i * e_l + g_l * e_i) * (g_j * e_k + g_k * e_j) + (g_j * e_l + g_l * e_j) * (g_i * e_k + g_k * e_i)

Phi_1 = Phi_1 / 2

print((Phi_0 * Phi_1.expand()))