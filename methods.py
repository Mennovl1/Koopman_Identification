import numpy
import System
import utils
import Dictionaries
import plot_data
import scipy.signal
import matplotlib.pyplot as plt
import time

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


def compute_eigenpairs_Korda2020_custom(H, N_g, rho = 0.1):
    '''Compute eigenfunction-eigenvalue pairs from measurement data H in shape N, N_t, N_n.
    
    Input: 
        H: N_n x N x N_t        Measurement data of N_t trajectories of length N with N_n measurements
        N_g:                    Number of eigenfunction/eigenvector pairs per measurement variable.
        rho:                    Regularisation parameter rho, default = 0.1

    Returns:
        L: N_n x N_g            Eigenvalues associated to the measurement data
        G: N_n x N_g x N_t      Initial conditions associated to the eigenvalues and data
        V: N_n x N   x N_g      Vandermonde matrix that is later used in Korda optimisation    
    '''
    N_n, N, N_t = H.shape

    print("Computing linear evolution in data to find eigenvalues and eigenfunctions on nonrecurrent set")
    start_time = time.perf_counter()
    # Compute linear evolution matrices from singular value decomposition
    U, _, _ = numpy.linalg.svd(H, full_matrices=False)

    # Compute eigenvalues associated to the evolution. s is shape N_n x N_g
    L, _ = numpy.linalg.eig( numpy.linalg.pinv(U[:, :N-1, :N_g]) @ U[:, 1: N, :N_g])

    # Compute vanderMonde matrix to find linear evolution
    V = numpy.empty((N_n, N, N_g), dtype=complex)
    for i in range(N_n):
        V[i,:,:] = numpy.vander(L[i,:], N, increasing=True).T
    
    # Compute initial conditions as regularised least-square solution
    G = numpy.linalg.solve( V.transpose((0,2,1)) @ V + rho * numpy.eye(N_g) , V.transpose((0,2,1)) @ H )

    elapsed = (time.perf_counter() - start_time) * 1000.0
    print(f"Computed eigenfunction, eigenvalue pairs in {elapsed:.1f} ms")
    return L, G, V


def compute_Linear_Evolution_Korda2020(H, N_g):
    N_n, N, N_t = H.shape

    print("Computing linear evolution in data to find initial states on nonrecurrent set")
    start_time = time.perf_counter()

    U, s, V = numpy.linalg.svd(H, full_matrices=False)

    Sigma = numpy.empty((N_n, N_g, N_g))

    for i in range(N_n):
        Sigma[i,:,:] = numpy.diag(s[i,:N_g])
    
    A = numpy.linalg.pinv(U[:, :N-1, :N_g]) @ U[:, 1: N, :N_g]
    # Perform diagonalisation
    L, P = numpy.linalg.eig( A )

    C = U[:, 0, :N_g] @ P

    G = numpy.linalg.inv(P) @ Sigma @ V[:, :N_g, :N_g].transpose(0,2,1)
    
    elapsed = (time.perf_counter() - start_time) * 1000.0
    print(f"Computed eigenfunction, eigenvalue pairs in {elapsed:.1f} ms")
    return L, C, G

def compute_Korda2020(X, N_g, f, rho=0.1):
    '''Compute eigenfunction based on eigenvalues L and initial function values g_0 on the nonrecurrent surface using the function library f
    
    
    Input:
        X: N_t x n x N          Data input matrix from autonomous system evolution
        N_g:                    Number of eigenvalue/eigenvector pairs used per measurement variable (so total n x N_g initial eigenfunctions)
        f:                      Function dictionary used to estimate linear evolution surface


    Output: 
        A:                      Linear evolution matrix, diag(L)
        C:                      Observation matrix to recover the original output
        optimal_lift:           Lifting function input
    '''
    N_t, n, N = X.shape
    # Reshape data matrix into appropriate format and compute eigenvector, eigenvalue pairs
    H = numpy.transpose(X, [1, 2, 0])
    # L, G, V = compute_eigenpairs_Korda2020_custom(H, N_g, rho=rho)
    L, C_obs, G = compute_Linear_Evolution_Korda2020(H, N_g)#, rho=rho)


    V = numpy.empty((n, N, N_g), dtype=complex)
    
    for i in range(n):
        V[i,:,:] = numpy.vander(L[i,...], N, increasing=True).T


    Z = numpy.empty(( f.size, N, N_t ))

    X_reshaped = X.reshape((n, -1))     # Flattens trajectory data into horizontally stacked data
    
    A = numpy.diag(L.flatten())


    print("Applying function dictionary on trajectory data")
    # for i in range(N_t):
    Z = utils.ApplyFunctionDictionary(X_reshaped, f)

    # Z_reshaped = Z.reshape((f.size, -1))
    # print(Z_reshaped.shape)

    # V is of n x N x N_g
    # G is of n x N_g x N_t
    # Solve problem per eigenfunction
    print("Starting computation of linear evolution surface from data")
    start_time = time.perf_counter()

    C = numpy.empty((n * N_g, f.size), dtype=complex)    
    for j in range(n):
        for i in range(N_g):

            G_ev = numpy.outer(V[j, :, i], G[j, i, :]).T.flatten()      # N x N_t -> vertical time evolution, horizontal trajectories, stack trajectories horizontally

            # Compute estimated eigenfunction, store such that row corresponds to a eigenvector
            C[j * N_g + i, :] = numpy.linalg.solve( Z @ Z.T + rho * numpy.eye(f.size), Z @ G_ev.T )



    elapsed = (time.perf_counter() - start_time) * 1000.0
    print(f"Computed eigenfunction surface in {elapsed:.1f} ms")

    def optimal_lift(X):
        return C @ utils.ApplyFunctionDictionary(X, f)

    # A = numpy.diag(L.flatten())
    # C_obs = numpy.kron(numpy.eye(n), numpy.ones((1, N_g)))

    return A, C_obs, optimal_lift


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



def linear_Validate(A, B, C, data_system, function_dictionary, time_delays):


    validation_data = data_system.validationData()
    validation_input = data_system.validationData(Data=data_system.InputSequence)
    N_sequences, n, N_length = validation_data.shape

    variance_accounted_for = numpy.empty((N_sequences, 2))
    root_mean_square = numpy.empty((N_sequences))
    rroot_mean_square = numpy.empty((N_sequences))

    for i in range(N_sequences):
        # Apply time-delays and lift coordinates
        Y_validate = utils.MakeHankelMatrix(validation_data[i,...], time_delays)

        # Compute prediction
        x0 = function_dictionary.apply(Y_validate[:,0].reshape((n, 1)) )
        t_out, Z_est, X_est = utils.linearpredict(x0, validation_input[i,...], A, B)
        
        Y_est = C @ Z_est.T #function_dictionary.recover(Z_est.T)

        variance_accounted_for[i], root_mean_square[i], rroot_mean_square[i] = utils.scoredata(Y_validate, Y_est)


    return variance_accounted_for, root_mean_square, rroot_mean_square


def linear_compare(A, B, C, system, function_dictionary, time_delays, x0, u):
    m,N = u.shape
    n, = x0.shape

    # Compute linear prediction
    # z0 = function_dictionary.apply(x0.reshape(n,1))
    z0 = function_dictionary(x0.reshape(n,1))

    t_out, z_est, x_est = utils.linearpredict(z0, u, A, B)
    # y_est = function_dictionary.recover(z_est.T)
    y_est = 0
    y_est2 = C @ z_est.T
    
    # Compute normal evolution
    t_comp, y_comp = system.evolve(x0, u)

    return t_out, y_est, y_comp, y_est2







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