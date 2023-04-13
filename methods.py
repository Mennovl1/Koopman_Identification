import numpy
import System
import utils
import Dictionaries
import plot_data
import scipy.signal
import scipy.sparse
import matplotlib.pyplot as plt
import time
import scipy.interpolate
# from progress.bar import Bar
from tqdm.notebook import trange, tqdm


def compute_DMD_solution(X_shift, X):
    '''Perform the core Dynamic mode Decomposition step on data matrices X shift and X'''
    U, S, V = numpy.linalg.svd(X)
    A = U.H @ X_shift @ V @ numpy.linalg.inv(S)
    return A


def compute_eDMD(X_shift, X, f):
    '''Perform the Dynamic mode Decomposition step on lifted state data'''
    Z_shift = utils.ApplyFunctionDictionary(X_shift, f)
    Z       = utils.ApplyFunctionDictionary(X, f)

    return compute_DMD_solution(Z_shift, Z)


def compute_Korda2018(X_shift, X, U, f):
    '''Compute the solution by Korda used in their 2018 paper for controlled systems'''
    n, N = X.shape

    # Lift the functions to increase state dimension
    Z_shift = utils.ApplyFunctionDictionary(X_shift, f)
    Z       = utils.ApplyFunctionDictionary(X, f)
    
    t = numpy.linspace(0, N * 0.01, N)
    
    # Prepare matrices for normal equation solutions for the least squares optimisation
    # Vvt = [ Y Y'   Y U' ]      WVt = [ Y+ Y'    Y+ U  ]
    #       [ U Y'   U U' ]            [ X  Y'    X  U' ]  
    W = numpy.vstack((Z_shift, X))
    V = numpy.vstack((Z, U))
    VVt = V @ V.T
    WVt = W @ V.T
    
    # Compute least square solution
    M = WVt @ numpy.linalg.pinv(VVt)

    # Extract optimised solution, M = [ A B ]
    #                                 [ C 0 ]
    A = M[0:f.size,0:f.size]
    B = M[0:f.size, f.size:]
    C = M[f.size:, :f.size]
    
    return A, B, C

def compute_Linear_Evolution_Korda2020(H, N_g, s = 10):
    '''Compute linear evolution estimate for eigenvalues and continuous function estimates on nonrecurrent set. 
    Comes down to decomposing H into a product of a Vandermonde matrix L, H = L G

    Args:
        H:          n x N x N_t array, n measurement variables measured for N_t sequences for length N. 
        N_g:        Number of eigenvalues in Vandermonde matrix per measurement variable
    
    Returns:
        L:          n x N x N_g array, Vandermonde matrix in decomposition
        G:          n x N_g x N_t array, Function estimates g_i(x(0)) on nonrecurrent set
        eigVals:    N_g array, Largest N_g eigenvalues associated to the system evolution
    '''

    print("Computing linear evolution in data to find initial states on nonrecurrent set")
    N_n, N, N_t = H.shape
    start_time = time.perf_counter()

    X = H.transpose((2,0,1))
    
    # Reformat into time-delay coordinates
    Y = numpy.empty((s+1, N-s, N_t))
    Data = numpy.empty((N_n, s+1, (N-s)*N_t))
    print(f"Formatting time-delay coordinates with delay s = {s}")
    for k in trange(N_n):
        for i in trange(N_t):
            Y[:,:,i] = utils.MakeHankelMatrix(X[i,k,:].reshape((1,N)), s)
        Data[k,:,:] = Y.reshape((s+1, -1), order='F')       # s+1 x (N-s)N_t

    u, _, _ = numpy.linalg.svd(Data, full_matrices=False, compute_uv=True)
    
    # Compute through svd subspace and associated shift-eigenvalues
    Phi = numpy.linalg.pinv(u[:,:-1, :N_g]) @ u[:,1:,:N_g]
    E = numpy.linalg.eigvals( Phi )
    
    eigVals = E #numpy.empty((N_n, N_g), dtype=complex)
    L = numpy.empty((N_n, N, N_g), dtype = complex)
    G = numpy.empty((N_n, N_g, N_t), dtype=complex)

    # Construct associated Vandermonde matrix from eigenvalues
    for i in range(N_n):
        L[i,...]     = numpy.vander(eigVals[i,:], N, increasing=True).T 
    
    # Compute initial function surface
    G = numpy.linalg.pinv(L) @ H

    # Report residuals of least square problem
    utils.report_Residuals(H, L @ G)
    
    elapsed = (time.perf_counter() - start_time) * 1000.0
    print(f"Computed eigenfunction, eigenvalue pairs in {elapsed:.1f} ms")
    return L, G, eigVals

def compute_Korda2020(X, N_g, rho=0.05, s=10, method="NN"):
    '''Compute eigenfunction based on eigenvalues L and initial function values g_0 on the nonrecurrent surface using the function library f
    
    
    Args:
        X: N_t x n x N          Data input matrix from autonomous system evolution
        N_g:                    Number of eigenvalue/eigenvector pairs used per measurement variable (so total n x N_g initial eigenfunctions)
        f:                      Function dictionary used to estimate linear evolution surface


    Returns: 
        A:                      Linear evolution matrix, diag(L)
        C:                      Observation matrix to recover the original output
        optimal_lift:           Lifting function input
    '''
    N_t, n, N = X.shape

    # Reshape data matrix into format n x N x N_t, compute decomposition
    H = numpy.transpose(X, [1, 2, 0])
    L, G, eig = compute_Linear_Evolution_Korda2020(H, N_g, s=s)
    # L is of n x N x N_g
    # G is of n x N_g x N_t
    
    # Flatten trajectory data into horizontally stacked data, i.e. n x N * N_t
    X_reshaped = H.reshape((n, -1), order='F')

    # Solve problem per eigenfunction
    print("Starting computation of linear evolution surface from data")
    start_time = time.perf_counter()

    interpolator = Dictionaries.Korda_Interpolator(method, X_reshaped, N_g, rho=rho)

    Compute_Eigensurface(N_g, n, L, G, interpolator)

    elapsed = (time.perf_counter() - start_time) * 1000.0
    print(f"Computed eigenfunction surface in {elapsed:.1f} ms")

    # Construct eigenfunction surface
    A = numpy.diag(eig.flatten())
 
    return A, interpolator


def Compute_Eigensurface(N_g, n, L, G, interpolator):
    '''Compute observable with linear evolution from linear evolution data
    
    Args:
        N_g:            Number of eigenfunctions associated to a measurement n
        n:              Measurement dimension
        L:              Vandermonde matrix defining the linear evolution of the surface
        G:              Initial states on nonrecurrent set associated to eigenvalues in Vandermonde matrix
    '''
    for j in trange(n):
        for i in trange(N_g):
            # Outer product between 1, l^1, l^2, l^3, ... and g1(x1) g1(x2) g1(x3) ...
            # Without transpose and without flatten, G_ev is N x N_t
            G_ev = numpy.outer(L[j, :, i], G[j, i, :]).T.flatten()      # N x N_t -> vertical time evolution, horizontal trajectories, stack trajectories horizontally
            # G_ev flattened is N*N_t
            interpolator.Add_Function(G_ev, i, j)
    return

def testControlledMethod(method, data_system, function_dictionary, time_delays):
    '''Perform control-compatible DMD method with data in data_system and given function_dictionary'''
    Y_shift_train, Y_train = utils.DataToMatrix(data_system.trainingData())
    

    _, U_train = utils.DataToMatrix(data_system.trainingData(Data=data_system.InputSequence))
    

    # Apply time-delayed coordinates
    Y_shift_train = utils.MakeHankelMatrix(Y_shift_train, time_delays)
    Y_train = utils.MakeHankelMatrix(Y_train, time_delays)
    
    p1 = data_system.trainingData(Data=data_system.InputSequence)
    # Compute controlled DMD estimate
    A, B, C = method(Y_shift_train, Y_train, U_train, function_dictionary)

    # Validate
    # vaf, rmse, rrmse = linear_Validate(A, B, C, data_system, function_dictionary, time_delays)
    return A, B, C



def linear_Validate(A, B, data_system, function_dictionary, time_delays, C=None):


    validation_data = data_system.validationData()
    validation_input = data_system.validationData(Data=data_system.InputSequence)
    N_sequences, n, N_length = validation_data.shape

    variance_accounted_for = numpy.empty((N_sequences, 2))
    root_mean_square = numpy.empty((N_sequences))
    rroot_mean_square = numpy.empty((N_sequences))

    for i in trange(N_sequences):
        # Apply time-delays and lift coordinates
        Y_validate = utils.MakeHankelMatrix(validation_data[i,...], time_delays)

        # Compute prediction
        x0 = function_dictionary.apply(Y_validate[:,0].reshape((n, 1)) )
        t_out, Z_est = utils.linearpredict(x0, validation_input[i,...], A, B)
        if C is None:
            Y_est = function_dictionary.recover(Z_est.T)
        else:
            Y_est = C @ Z_est #

        variance_accounted_for[i], root_mean_square[i], rroot_mean_square[i] = utils.scoredata(Y_validate, Y_est)


    return variance_accounted_for, root_mean_square, rroot_mean_square


def linear_compare(A, B, system, function_dictionary, time_delays, x0, u, C=None):
    m,N = u.shape
    n, = x0.shape

    # Compute linear prediction
    print("Lifting dynamics")
    z0 = function_dictionary.apply(x0.reshape(n,1))

    t_out, z_est = utils.linearpredict(z0, u, A, B)
    y_est = function_dictionary.recover(z_est)

    t_comp, y_comp = system.evolve(x0.flatten(), u)
    if C is None:
        return t_out, y_est, y_comp
    else:
        return t_out, y_est, y_comp, C @ z_est







def main():
    KordaEquation                   = System.differentialEquation(System.forcedvanDerPolSystemKorda, 2, 1)
    Korda                           = System.System(KordaEquation, 'testdata/Korda2018.mat', 200, 1500, 0.01, Generate=False)


    tprbf_dict = Dictionaries.thin_plate_rbf_dictionary(2, 100, 2)

    A, B, C = testControlledMethod(compute_Korda2018, Korda, tprbf_dict, 1)
    vaf, rmse, rrmse = linear_Validate(A, B, C, Korda, tprbf_dict, 1)
    
    u_square = scipy.signal.square(numpy.linspace(0, 10 * numpy.pi, int(3 // Korda.dt))) * 2.0 - 1.0
    x0 = numpy.array([0.5, 0.5])

    t_out, y_est, y_comp, y_est2 = linear_compare(A, B, C, Korda, tprbf_dict, 1, x0, u_square.reshape((1,u_square.shape[0])))

    plot_data.compareTrajectory(t_out, y_comp, y_est, y_est2)

if __name__ == "__main__":
    main()