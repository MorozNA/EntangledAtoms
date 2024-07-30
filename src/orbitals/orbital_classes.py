import numpy as np
import copy
from IPython.display import display, Math
from itertools import permutations


class PrimitiveGaussian:
    def __init__(self, alpha: float, coeff: float, coordinates: np.array):
        self.alpha = alpha
        self.coeff = coeff
        self.coordinates = coordinates

    def to_print(self, index):
        to_print = r'{} \exp(-{} \cdot |{} - r_{}|^2)'.format(self.coeff, self.alpha, self.coordinates, index)
        display(Math(to_print))

    def __mul__(self, other):
        alpha = self.alpha + other.alpha
        coordinates = (self.alpha * self.coordinates + other.alpha * other.coordinates) / alpha
        s = self.alpha * np.dot(self.coordinates, self.coordinates) + other.alpha * np.dot(other.coordinates,
                                                                                           other.coordinates)
        coeff = self.coeff * other.coeff * np.exp(np.dot(coordinates, coordinates) * alpha - s)
        return PrimitiveGaussian(alpha, coeff, coordinates)

    def integrate(self):
        return self.coeff * np.sqrt(np.pi / self.alpha) ** 3

    def get_value(self, coordinates):
        distance2 = np.dot(coordinates - self.coordinates, coordinates - self.coordinates)
        value = self.coeff * np.exp(-self.alpha * distance2)
        return value


class AtomicOrbital:
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
        return AtomicOrbital(pg_list, el_ind)

    def integrate(self, electronic_ind):
        new_ao = copy.deepcopy(self)
        # TODO: don't allow same el_ind
        index = new_ao.electronic_ind.index(electronic_ind)
        value = new_ao.pg_list[index].integrate()
        new_ao.pg_list[0].coeff *= value
        # new_ao.coeff *= value

        new_ao.electronic_ind.pop(index)
        new_ao.pg_list.pop(index)
        return new_ao

    def integrate_orbital(self):
        result = self.get_coeff()
        for pg in self.pg_list:
            result *= pg.integrate()
        return result

    def get_value(self, electronic_coordinates: list):
        assert len(self.electronic_ind) == len(electronic_coordinates)
        value = 1
        for i in range(len(self.electronic_ind)):
            value *= self.pg_list[i].get_value(electronic_coordinates[self.electronic_ind[i]])
        return value


class MolecularOrbital:
    def __init__(self, ao_list: list[AtomicOrbital]):
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
        return MolecularOrbital(ao3_list)

    def __add__(self, other):
        ao_list = self.ao_list + other.ao_list
        return MolecularOrbital(ao_list)

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
        return MolecularOrbital(new_ao_list)

    def integrate_orbital(self):
        result = 0
        for ao in self.ao_list:
            ao.integrate_orbital()
            result += ao.get_coeff()
        return result

    def normalize(self):
        module = self * self.conj()
        K = module.integrate_orbital()
        for ao in self.ao_list:
            # TODO: divide coeffs between pg's or do something else
            ao.pg_list[0].coeff = ao.pg_list[0].coeff / K

    # TODO: find similar terms and define addition of similar primitive_gaussians


def transpose_list(or_list):
    # Works for triangular lists
    new_list = []
    for col in range(len(or_list)):
        column = [or_list[i][col] for i in range(len(or_list[col]))]
        new_list.append(column)
    return new_list


class YoungDiagram:
    def __init__(self, diagram: list, signs=None):
        # TODO: assert whether len(signs) = len(diagram)
        self.diagram = diagram
        if signs is None:
            signs = ['+'] * len(diagram)
        self.signs = signs

    def __str__(self):
        # str signs = : each 0 to '+', 1 to '-'
        for i in range(len(self.diagram[:-1])):
            print(self.diagram[i][0], self.signs[i + 1], '', end='')
        print(self.diagram[-1][0])

        for diagram in self.diagram:
            for row in diagram[1::]:
                num_spaces = len(self.diagram[0][0]) - len(row)
                print(row, num_spaces * '.  ', ' ', end='')
        return ''

    def symm_row(self, row):
        new_diagrams = []
        new_signs = []
        # for diagram in self.diagram:
        for i in range(len(self.diagram)):
            for perm in list(permutations(self.diagram[i][row]))[1::]:
                # neglecting first permutation which is the initial diagram
                symm_diagram = self.diagram[i].copy()
                symm_diagram[row] = list(perm)
                new_diagrams.append(symm_diagram)
                new_signs.append(self.signs[i])
        self.diagram = self.diagram + new_diagrams
        self.signs = self.signs + new_signs

    def antisymm_row(self, row):
        new_diagrams = []
        new_signs = []
        for i in range(len(self.diagram)):
            for perm in list(permutations(self.diagram[i][row]))[1::]:
                # neglecting first permutation which is the initial diagram
                symm_diagram = self.diagram[i].copy()
                symm_diagram[row] = list(perm)
                new_diagrams.append(symm_diagram)
                if self.signs[i] == '+':
                    new_signs.append('-')
                elif self.signs[i] == '-':
                    new_signs.append('+')
        self.diagram = self.diagram + new_diagrams
        self.signs = self.signs + new_signs

    def symm_col(self, col):
        new_diagrams = []
        new_signs = []
        for i in range(len(self.diagram)):
            transposed_diagram = transpose_list(self.diagram[i])
            for perm in list(permutations(transposed_diagram[col]))[1::]:
                # neglecting first permutation which is the initial diagram
                symm_diagram = transposed_diagram.copy()
                symm_diagram[col] = list(perm)
                new_diagrams.append(transpose_list((symm_diagram)))
                new_signs.append(self.signs[i])
        self.diagram = self.diagram + new_diagrams
        self.signs = self.signs + new_signs

    def antisymm_col(self, col):
        new_diagrams = []
        new_signs = []
        for i in range(len(self.diagram)):
            transposed_diagram = transpose_list(self.diagram[i])
            for perm in list(permutations(transposed_diagram[col]))[1::]:
                # neglecting first permutation which is the initial diagram
                symm_diagram = transposed_diagram.copy()
                symm_diagram[col] = list(perm)
                new_diagrams.append(transpose_list((symm_diagram)))
                if self.signs[i] == '+':
                    new_signs.append('-')
                elif self.signs[i] == '-':
                    new_signs.append('+')
        self.diagram = self.diagram + new_diagrams
        self.signs = self.signs + new_signs

    def get_orbital(self, pg_list):
        for num in range(len(self.diagram)):
            flatten_diagram = [x for xs in self.diagram[num] for x in xs]
            ao = AtomicOrbital(pg_list, flatten_diagram)
            # sign = int((self.signs[num] + '1').replace('"', ''))
            if num == 0:
                mo = MolecularOrbital([ao])
            else:
                if self.signs[num]=='+':
                    mo = mo + MolecularOrbital([ao])
                elif self.signs[num]=='-':
                    mo = mo - MolecularOrbital([ao])
        return mo
