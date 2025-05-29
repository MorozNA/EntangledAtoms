import numpy as np
import copy
from IPython.display import display, Math


class PrimitiveGaussian:
    def __init__(self, alpha: np.array, coeff: float, coordinates: np.array):
        assert len(alpha) == len(coordinates)
        self.alpha = alpha
        self.coeff = coeff
        self.coordinates = coordinates

    def to_print(self, index):
        to_print = r'{} \exp(-{} \cdot |{} - r_{}|^2)'.format(self.coeff, self.alpha, self.coordinates, index)
        display(Math(to_print))

    def __mul__(self, other):
        alpha = self.alpha + other.alpha
        coordinates = (self.alpha * self.coordinates + other.alpha * other.coordinates) / alpha
        s = self.alpha * self.coordinates ** 2 + other.alpha * other.coordinates ** 2
        coeff = self.coeff * other.coeff * np.prod(np.exp(coordinates ** 2 * alpha - s))
        return PrimitiveGaussian(alpha, coeff, coordinates)

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return PrimitiveGaussian(self.alpha, other * self.coeff, self.coordinates)

    def integrate(self):
        return self.coeff * np.prod(np.sqrt(np.pi / self.alpha))

    def get_value(self, coordinates):
        distance2 = (coordinates - self.coordinates) ** 2
        value = self.coeff * np.exp(-np.dot(self.alpha, distance2))
        return value

    def get_values(self, r1, r2):
        distance = np.array([r1 - self.coordinates[0], r2 - self.coordinates[1]])
        distance = distance ** 2
        # TODO: use np.dot
        prod = self.alpha[0] * distance[0] + self.alpha[1] * distance[1]
        return self.coeff * np.exp(-prod)


class GaussianProduct:
    def __init__(self, pg_list: list[PrimitiveGaussian], electronic_ind: list):
        # TODO: copy deepcopy so i can change coeffs individually between AtomicOrbitals
        self.pg_list = copy.deepcopy(pg_list)
        self.electronic_ind = electronic_ind

    def __str__(self):
        to_print = ''
        for i in range(len(self.pg_list) - 1):
            to_print += r'\phi_{} ({}) \; \cdot'.format(i, self.electronic_ind[i])
        to_print += r'\phi_{} ({})'.format(len(self.pg_list) - 1, self.electronic_ind[-1])
        display(Math(to_print))
        return ''

    def get_coeff(self):
        coeff = 1
        for pg in self.pg_list:
            coeff *= pg.coeff
        return coeff

    def __mul__(self, other):
        # TODO: don't allow same el_ind
        # TODO: assert if all self.el_ind are the same as other.el_ind
        ao1 = copy.deepcopy(self)
        ao2 = copy.deepcopy(other)
        index_j = []
        pg_list = []
        el_ind = []
        # TODO: improve: better algrotithm to delete elements from the end of pg_list
        for i in reversed(range(len(self.pg_list))):
            for j in range(len(other.pg_list)):
                if self.electronic_ind[i] == other.electronic_ind[j]:
                    pg_list.append(self.pg_list[i] * other.pg_list[j])
                    el_ind.append(self.electronic_ind[i])
                    ao1.pg_list.pop(i)
                    ao1.electronic_ind.pop(i)
                    index_j.append(j)
                    # ao2.pg_list.pop(j)
                    # ao2.electronic_ind.pop(j)
        index_j.sort(reverse=True)
        for j in index_j:
            ao2.pg_list.pop(j)
            ao2.electronic_ind.pop(j)
        pg_list = pg_list + ao1.pg_list + ao2.pg_list
        el_ind = el_ind + ao1.electronic_ind + ao2.electronic_ind
        return GaussianProduct(pg_list, el_ind)

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            new_ao = copy.deepcopy(self)
            new_ao.pg_list[0] = other * new_ao.pg_list[0]
            return new_ao

    def integrate(self, electronic_ind):
        new_ao = copy.deepcopy(self)
        # TODO: don't allow same el_ind
        index = new_ao.electronic_ind.index(electronic_ind)
        value = new_ao.pg_list[index].integrate()
        new_ao.electronic_ind.pop(index)
        new_ao.pg_list.pop(index)

        new_ao.pg_list[0].coeff *= value
        # new_ao.coeff *= value
        return new_ao

    def integrate_orbital(self):
        result = 1
        for pg in self.pg_list:
            result *= pg.integrate()
        return result

    def get_value(self, electronic_coordinates: list):
        assert len(self.electronic_ind) == len(electronic_coordinates)
        value = 1
        for i in range(len(self.electronic_ind)):
            value *= self.pg_list[i].get_value(electronic_coordinates[self.electronic_ind[i]])
        return value

    def get_values(self, r1: list, r2: list):
        assert len(r1) == len(r2)
        assert len(self.electronic_ind) == len(r1)
        values = np.ones_like(r1)
        for i in range(len(self.electronic_ind)):
            values *= self.pg_list[i].get_values(r1[self.electronic_ind[i]], r2[self.electronic_ind[i]])


class Orbital:
    def __init__(self, ao_list: list[GaussianProduct]):
        self.ao_list = ao_list

    def __str__(self):
        to_print = ''
        for ao in self.ao_list:
            if ao.get_coeff() > 0:
                to_print += '+'
            else:
                to_print += '-'
            for i in range(len(ao.pg_list) - 1):
                to_print += r'\phi_{} ({}) \; \cdot'.format(i, ao.electronic_ind[i])
            to_print += r'\phi_{} ({})'.format(len(ao.pg_list) - 1, ao.electronic_ind[-1])
        to_print = to_print[1:]
        display(Math(to_print))
        return ''

    def __mul__(self, other):
        ao3_list = []
        for ao1 in self.ao_list:
            for ao2 in other.ao_list:
                ao3_list.append(ao1 * ao2)
        return Orbital(ao3_list)

    def __rmul__(self, other: float):
        if isinstance(other, float) or isinstance(other, int):
            new_orbital = copy.deepcopy(self)
            for i in range(len(new_orbital.ao_list)):
                new_orbital.ao_list[i] = other * new_orbital.ao_list[i]
            return new_orbital

    def __pow__(self, other: int):
        new_orbital = copy.deepcopy(self)
        if other == 0:
            return 1
        else:
            for _ in range(other - 1):
                new_orbital = new_orbital * self
            return new_orbital

    def __add__(self, other):
        ao_list = self.ao_list + other.ao_list
        return Orbital(ao_list)

    def __sub__(self, other):
        # TODO: check wheter 'other' changes after applying this method
        # TODO: change every first coeff for pg in ao
        # TODO: all ao_list's are linked with one pg_list, but coeff should be changed only in 1 list
        to_sub = copy.deepcopy(other)
        for ao in to_sub.ao_list:
            ao.pg_list[0].coeff = -ao.pg_list[0].coeff
        return self + to_sub

    def conj(self):
        # TODO: redefine coeff to complex value
        return self

    def get_value(self, electronic_coordinates: list):
        result = 0
        for ao in self.ao_list:
            result += ao.get_value(electronic_coordinates)
        return result

    def integrate(self, electronic_index):
        new_ao_list = []
        for ao in self.ao_list:
            new_ao_list.append(ao.integrate(electronic_index))
        return Orbital(new_ao_list)

    def integrate_orbital(self):
        result = 0
        for ao in self.ao_list:
            result += ao.integrate_orbital()
        return result

    def normalize(self):
        module = self * self.conj()
        K = module.integrate_orbital()

        if K != 0:
            for ao in self.ao_list:
                ao.pg_list[0].coeff = ao.pg_list[0].coeff / np.sqrt(K)
