import matplotlib.pyplot as plt



def compareTrajectories(t, *args, **kwargs):
    
    
    n, N = args[0].shape

    for i in range(n):
        plt.figure()
        for Y in args:
            plt.plot(t, Y[i,:])
        plt.grid()
        plt.legend(kwargs["Names"])
        plt.xlabel("t (s)")
        plt.ylabel('x' + str(i))
        plt.show(block=False)
    plt.show()