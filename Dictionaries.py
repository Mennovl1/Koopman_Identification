import numpy
from sympy.utilities.iterables import multiset_permutations
from utils import partitions
import scipy.interpolate
import utils

class Dict_Wrapper:
    def __init__(self):
        return

    
class monomial_dictionary(Dict_Wrapper):
    '''Class representing a monomial function dictionary that can be applied to data matrices for DMD methods'''
    
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.generate_function(n,d)


    def add_partitions_up_to(self, degree, permutation_list):
        '''Generate all permutations of a given degree with less then n components and append to list permutations_list'''
        for i in partitions(degree):
            if len(i) <= self.n:
                partition_array = numpy.zeros((self.n))
                partition_array[0:len(i)] = i

                for powers in multiset_permutations(partition_array):
                    permutation_list.append(powers)

    def generate_function(self, n, d):
        '''Generate a function for monomials up to degree d, with state dimension n'''
        permutation_list = []
        # Loop over 
        for degree in range(1,d+1):
            # Compute partitions of given degree
            self.add_partitions_up_to(degree, permutation_list)
        
        self.size = len(permutation_list)
        self.exponents = numpy.asarray(permutation_list)

    def apply(self, X):
        '''Apply generated monomial vector function to data matrix of format n x N'''
        if len(X.shape) > 1:
            n,N = X.shape
        else:
            n = X.shape
            N = 1
        Z = numpy.empty((self.size, N))

        for i in range(self.size):
            # Extract row, and apply the power permutations to this row
            Z[i,:] = numpy.prod(numpy.power(X, self.exponents[i,:].reshape((n,1))), axis=0)
        return Z

    def recover(self, X):
        '''Recover original state from lifted coordinates'''
        # Extract first n elements from data input
        return X[0:self.n, :]

    def __str__(self):
        '''Print monomial vector effectively'''
        text_out = "" #str(self.exponents)
        for i in self.exponents:
            for j in range(self.n-1):
                if i[j] != 0:
                    text_out = text_out + "(x" + str(j+1) + ")^" + str(int(i[j])) + " * "
            if i[self.n-1] != 0:
                text_out = text_out + "(x" + str(self.n) + ")^" + str(int(i[self.n-1])) + "\n"
            else:
                text_out = text_out[:-2] + "\n"
        return text_out



class thin_plate_rbf_dictionary(Dict_Wrapper):
    '''Thin plate radial basis function dictionary. || x - x0 ||^2 log(|| x - x0 ||)'''

    def __init__(self, n, p, size):
        self.n = n
        self.p = p
        self.size = p + n

        self.generate_function(size)

    def generate_function(self, size):
        self.origins = numpy.random.rand(self.p , self.n, 1) * size - size / 2.0
        # self.origins = numpy.zeros((self.p, self.n, 1))


    def apply(self, X):
        # if len(X.shape) > 1:
        #     n, N = X.shape
        # else:
        #     n = X.shape
        distance = self.origins - numpy.tile(X, (self.p, 1, 1))


        norm = numpy.linalg.norm(distance, axis=1)
        return numpy.vstack((X, numpy.square(norm) * numpy.log(norm)))



    def recover(self, X):
        return X[0:self.n, :]


class Korda_Interpolator(Dict_Wrapper):
    '''Linear interpolator constructed for usage with Korda's method for eigensurface computation'''

    def __init__(self, method, X, N_g, rho = 0.05):
        self.n, _ = X.shape
        self.method = method
        self.measurements = X.T   # N * N_t x n state measurement variable
        self.N_g = N_g
        self.rho = rho
        self.size = N_g * self.n
        self.C = numpy.kron(numpy.eye(self.n), numpy.ones((1, N_g)))

        if method == "NN" or method == "Linear":
            self.f = []
        elif isinstance(method, Dict_Wrapper):
            print("Lifting measurements")
            self.f = numpy.empty((n * N_g, method.size), dtype=complex)
            self.measurements = utils.ApplyFunctionDictionary(X, method)
        else:
            raise Exception("Undefined interpolation method")
    

    def Add_Function(self, Y, i,j):
        if self.method == "NN":
            self.f.append(scipy.interpolate.NearestNDInterpolator(self.measurements, Y))
        elif self.method == "Linear":
            self.f.append(scipy.interpolate.LinearNDInterpolator(self.measurements, Y))
        else:
            # Solve Z.T c - G_ev.T => c.T Z = G_ev (horizontally stacked)
            self.f[j*self.N_g + i, :] = numpy.linalg.solve( self.measurements @ self.measurements.T + self.rho * numpy.eye(self.method.size), self.measurements @ Y.T)

    def apply(self, X):
        if self.method == "NN" or self.method == "Linear":
            res = []
            for lerp_opt in self.f:
                res.append(lerp_opt(X.T))
            return numpy.asarray(res)
        else:
            return self.f @ self.method.apply(X)
    
    def recover(self, X):
        return self.C @ X


for_testing = thin_plate_rbf_dictionary(4, 2, 2)

# X = numpy.array([[0.0, 0,0,0], [0.0,0,0.0,1]])
X = numpy.random.rand(4, 100) * 2 - 1.0

print(for_testing.origins)

distance = for_testing.origins - X
norm = numpy.linalg.norm(distance, axis=1)
print(norm)

res = for_testing.apply(X)

# diff = res - X[0:4]

print(for_testing.apply(X))



# test_func = thin_plate_rbf_dictionary(2, 3, 2)

# X = numpy.array([[1.0, 0,0,0], [1.0,0,1.0,0]])

# test_func.apply(X)


# test_func = monomial_vector(4, 2, 3)