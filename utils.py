import numpy
import scipy.signal


def DataToMatrix(X):
    '''Format Data matrix of multiple state or measurement sequences into a pair of data matrices X and X+, the one-time step shifted version'''
    # Extract dimensions
    N_seq, n, N_length = numpy.shape(X)

    # Perform manipulation so column order stacking occurs
    X_shift = numpy.transpose(X[:,:,1:], (1, 2, 0))
    X_new = numpy.transpose(X[:,:,:-1], (1,2,0))
    matrix_shift = numpy.reshape(X_shift, (n, N_seq * (N_length-1)), order='F')
    matrix = numpy.reshape(X_new, (n, N_seq * (N_length-1)), order='F')

    # Output in order X+, X. "Left shift operation on X"
    return matrix_shift, matrix


def ApplyFunctionDictionary(X, f):
    '''Apply data dictionary given by vector function output f on data matrix, expanding state dimension'''
    return f.apply(X)

def MakeHankelMatrix(X, s):
    '''Turn data matrix into hankel matrix with time delay coordinates, implementation taken from https://stackoverflow.com/questions/71410927/vectorized-way-to-construct-a-block-hankel-matrix-in-numpy-or-scipy '''
    d = s+1
    n, N = X.shape
    if s > 0:
        return numpy.lib.stride_tricks.sliding_window_view(X, (n, N+1-d)).reshape(d*n, -1)
    else:
        return X


def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p


def linearpredict(X0, U, A, B, dt=0.01):
    n,_ = X0.shape
    m,N = U.shape

    t_out = numpy.linspace(0, (N-1) * dt, N, endpoint=True)
    x_out = numpy.empty((n, N), dtype=complex)
    x_out[...,0] = X0.flatten()
    for i in range(N-1):
        x_out[...,i+1] = A @ x_out[...,i] + B @ U[:,i]

    # sp_system = scipy.signal.dlti(A, B, numpy.eye(n), numpy.zeros((n,m)), dt=0.01)
    # print(X0.flatten().ndim)
    return t_out, x_out #scipy.signal.dlsim(sp_system, U.T, x0=X0.flatten())

def VAF(Y, Yhat):
    '''Compute the Variance Accounted For row-wise (n x N) data length'''
    return 1 - numpy.var(Y - Yhat, axis=1) / numpy.var(Y, axis=1)

def RMSE(Y, Yhat):
    '''Compute the Root Mean Square Error for row-wise (n x N) data length'''
    return numpy.sqrt(numpy.linalg.norm(numpy.var(Y - Yhat, axis=1)))

def RRMSE(Y, Yhat):
    '''Compute the relative Root Mean Square Error for row-wise (n x N) data lenght'''
    return RMSE(Y, Yhat) / numpy.sqrt(numpy.linalg.norm(numpy.var(Y, axis=1)))


def scoredata(Y, Yhat):
    a = VAF(Y, Yhat)
    b = RMSE(Y,Yhat)
    c =RRMSE(Y, Yhat)
    return a, b, c 


def report_Residuals(X, Y):
    n, N, Nt = X.shape

    for i in range(n):
        res = 100 * numpy.linalg.norm(X[i,...] - Y[i,...],2) / numpy.linalg.norm(X[i,...],2)
        print(f'Found relative residual in x{i} of {res:.3f} %')
    print(numpy.max(X[i,...] - Y[i,...]))

def main():
    n = 2
    N_seq = 3
    N_length = 4
    X = numpy.reshape(numpy.linspace(0, 23, 24), (N_seq, n, N_length))
    print("Sequence 1: ")
    print(X[0,:,:])
    print("Sequence 2: ")
    print(X[1,:,:])
    print("Sequence 3: ")
    print(X[2,:,:])



    print("Constructing data matrices")
    X_shift, X_data = DataToMatrix(X)
    print(X_data)
    print(X_shift)

    print("Evolved sequence:")
    print(MakeHankelMatrix(X_data, 5))
    print(MakeHankelMatrix(X_shift,5))

if __name__ == "__main__":
    main()



