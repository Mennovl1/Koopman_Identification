import numpy
def integrate_RK(fun, x0, u, dt):
    n, = x0.shape
    N, = u.shape
    t = numpy.linspace(0, N // dt, N)
    
    f = lambda t, x, u : numpy.asarray(fun(t,x,u))
    
    
    # Beun RK4 method for testing, format for f(t, x, u)
    k1 = lambda t, x, u : f(t,x,u)
    k2 = lambda t, x, u : f(t, x + k1(t, x, u) * dt / 2.0, u)
    k3 = lambda t, x, u : f(t, x + k2(t, x, u) * dt / 2.0, u)
    k4 = lambda t, x, u : f(t, x + k1(t, x, u) * dt, u) 

    f_ud = lambda t, x, u : x + (dt / 6.0) * (k1(t,x,u) + 2 * k2(t,x,u) + 2 * k3(t,x,u) + k4(t,x,u))

    y = numpy.empty((n,N))
    y[:,0] = x0
    for i in range(N-1):
        y[:,i+1] = f_ud(0, y[:,i], u[i])  
        # print(y)
    # print(y)
    return t, y