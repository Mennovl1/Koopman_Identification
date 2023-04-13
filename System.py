import os
import numpy
import utils
import scipy.integrate
import scipy.io

class System:
    '''Data and function handle class for systems applied to DMD methods
    
    Properties:
        EvolutionFunction:          X -> TX, continuous-time evolution function
        StateSequence:              Data set
        MeasurementFunction:        Output function
        MeasurementSequence:        (N_length x N_sequences) Data set of measurements
        n:                          Underlying state dimension
        m:                          Measurement state dimension
        N_length:                   Measurement/State sequence length
        N_sequences:                Number of state/measurement sequences
        HasControl:                 True/False
        hasMeasurement:             True/False'''
    
    def  __init__(self, f, path, N_length, N_sequences, dt, **kwargs):
        self.EvolutionFunction = f
        self.N_length  = N_length
        self.N_sequences = N_sequences
        self.path = path
        self.n = f.n
        self.k = f.k
        self.dt = dt
        self.T_final = (self.N_length - 1.0) * dt
        self.split              = [2,1]                         # Ratio between training and validation data
        self.Autonomous = False
        self.initfun = None
        GenerateNow = True

        if os.path.isfile(path):
            GenerateNow = False


        # Process keyword arguments
        for key, value in kwargs.items():
            if key == 'MeasurementFunction':
                self.HasMeasurement = True
                self.MeasurementFunction = value
            if key == "Autonomous" and value:
                self.Autonomous = True
            if key == 'Generate' and value:
                GenerateNow = True
            if key == "split":
                self.split = value
            if key == "Initialisation_function":
                self.initfun = value
            else:
                print("Key '" + key + "' not found in system construction")

        # If data does not exist, generate new data
        if GenerateNow:
            self.makeNewData()
        else:
            self.readData()
        return 

    def makeNewData(self):
        '''Generate new dataset and place data in mat file'''
        self.TimeSequence, self.StateSequence, self.InputSequence = self.generateData()

        data_dict = {"StateSequence": self.StateSequence,
                        "TimeSequence": self.TimeSequence,
                        "InputSequence": self.InputSequence,
                        "dt": self.dt,
                        "N_sequences": self.N_sequences,
                        "N_length": self.N_length,
                        "T_final": self.T_final,
                        "Autonomous": self.Autonomous}

        scipy.io.savemat(self.path, data_dict)
        return

    def readData(self):
        '''Extract data from existing dataset'''
        print("Using existing dataset at filepath")
        matData = scipy.io.loadmat(self.path)

        self.StateSequence      = matData["StateSequence"]      # Of shape [N_sequences, n, N_length]
        self.TimeSequence       = matData["TimeSequence"]
        self.InputSequence      = matData["InputSequence"]
        self.dt                 = matData["dt"].item()
        self.N_sequences        = matData["N_sequences"].item()
        self.N_length           = matData["N_length"].item()
        self.T_final            = matData["T_final"].item()
        self.Autonomous         = matData["Autonomous"].item()
        return  

    def generateData(self, initfun=None):
        '''Generate dataset from random initial conditions'''
        print("Generating new dataset")
        y = numpy.empty((self.N_sequences, self.n, self.N_length))

        t = numpy.linspace(0, self.T_final, self.N_length)                          # Associated time stamps

        x0 = self.initialConditions()
        u  = self.generateInput()

        for i in range(self.N_sequences):
            y[i,...] = scipy.integrate.solve_ivp(self.EvolutionFunction.evaluate, [0,self.T_final], x0[i,:], t_eval=t, args=[u[i,...], self.dt]).y
        return t,y,u

    def trainingData(self, split=None, Data = None):
        '''Return training data based on whether multiple trajectories were sampled or a single trajectory was sampled. Data is split in ratio 2:1 by default'''
        if Data is None:
            Data = self.StateSequence
        if split is not None:
            self.split = split
        if self.N_sequences > 1:
            print(self.N_sequences // sum(self.split) * self.split[0])
            return Data[0 : self.N_sequences // sum(self.split) * self.split[0], ...]
        else:
            return Data[..., 0 : self.N_length // sum(self.split) * self.split[0]]

    def validationData(self, split=None, Data = None):
        '''Return validation data based on whether multiple trajectories were sampled or a single trajectory was sampled. Data is split in ratio 2:1 by default'''
        if Data is None:
            Data = self.StateSequence
        if split is not None:
            self.split = split
        if self.N_sequences > 1:
            return Data[self.N_sequences - self.N_sequences // sum(self.split) * self.split[0] :, ...]
        else:
            return Data[..., self.N_length - self.N_length // sum(self.split) * self.split[0] : ]

    def convertData(self, Data = None):
        '''Convert data format into data matrices that can be used for DMD'''
        if Data is None:
            Data = self.StateSequence
        X = utils.DataToMatrix(Data)
        # U = utils.DataToMatrix(self.InputSequence)
        return X

    def evolve(self, x0, u):
        m, N = u.shape
        T_final = (N-1) * self.dt
        t = numpy.linspace(0, T_final, N)

        y = scipy.integrate.solve_ivp(self.EvolutionFunction.evaluate, [0,T_final], x0, t_eval=t, args=[u, self.dt]).y  
        
        return t, y

    def initialConditions(self):
        '''Generate initial conditions using specified initial condition function'''
        if self.initfun is None:
            return numpy.random.rand(self.N_sequences, self.n) * 2.0 - 1.0                # Generate random initial conditions
        else:
            return self.initfun(self.N_sequences, self.n)                                      # Generate initial conditions using external function

    def generateInput(self):
        '''Generate input sequence for (non)autonomous system'''
        if self.Autonomous:
            return numpy.zeros((self.N_sequences, self.k, self.N_length))                  # Generate zero input for autonomous input
        else:
            return numpy.random.rand(self.N_sequences, self.k, self.N_length) * 2.0 - 1.0  # Generate random white noise control signal

class differentialEquation:
    '''Class for representing differential equations, including associated properties when desired.'''

    def __init__(self, f, n, k, args=None):
        self.EvolutionFunction = f
        self.n = n
        self.k = k
        self.extra_args = args

    def evaluate(self, t, x, u, dt):
        if self.extra_args is None:
            return self.EvolutionFunction(t, x, u[..., int(t / dt)])
        else:
            return self.EvolutionFunction(t, x, u[..., int(t / dt)], self.extra_args)


# Different differential equations
def LorenzAttractorDifferential(time, state, control, sigma=10.0, beta=8.0/3.0, rho=28.0):
    '''Differential equation for the Lorenz attractor. Optional arguments sigma, rho and beta'''

    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def TuExampleSystem(time, state, control, mu=0.5, f_lambda=0.9):
    '''Simple differential equation for which analytic Koopman terms are known. Optional arguments: mu and lambda'''

    x,y = state
    return [f_lambda * x, mu * y + (f_lambda**2.0 - mu) * x]

def LinearSystem(time, state, control, A=None):
    '''Linear system. Optional arguments: matrix A, else 3x3 random'''
    if A is None:
        A = numpy.array([[1, 2, 3], [2, 4, 5], [1, 8, 1]])
        Astable = - A @ A.transpose()
        # print(numpy.linalg.eig(Astable))
        B = numpy.zeros((3,1))

    return Astable @ state + B @ control

def vanDerPolSystem(time, state, control, mu=10.0):
    '''Van der Pol oscillator. Optional arguments mu'''
    x,y = state
    return [y, mu * (1.0 - numpy.square(x)) * y - x]

def forcedvanDerPolSystem(time, state, control, mu = 10.0):
    '''Forced Van der Pol oscillator. Optional arguments mu'''
    x,y = state
    return [y, mu * (1.0 - numpy.square(x)) * y - x + control]

def PendulumSystem(time, state, control):
    '''Equation for a simple nonlinear pendulum'''
    x,y = state
    return [y, - numpy.sin(x)]


def forcedvanDerPolSystemKorda(time, state, control):
    '''Forced van der Pol oscillator with parameterisation according to Korda2018'''
    x,y = state
    return [2 * y, -0.8 * x + 2 * y - 10 * x * x * y + control]

# Generate test data to test functionality

def main():
    LorenzEquation                  = differentialEquation(LorenzAttractorDifferential, 3, 0)
    TuEquation                      = differentialEquation(TuExampleSystem, 2, 0)
    LinearEquation                  = differentialEquation(LinearSystem, 3, 1)
    VanDerPolEquation               = differentialEquation(vanDerPolSystem, 2, 0)
    PendulumEquation                = differentialEquation(PendulumSystem, 2, 0)
    controlledVanDerPolEquation     = differentialEquation(forcedvanDerPolSystem, 2, 1);


    Lorenz              = System(LorenzEquation, "testdata/test_path.mat", 500, 3, 0.05, Generate=False)
    Tu                  = System(TuEquation,     'testdata/tu_test.mat',    50, 50, 0.01, Generate=False)
    Linear              = System(LinearEquation, 'testdata/linear_test.mat', 200, 1, 0.05, Generate=False)
    VanDerPol           = System(VanDerPolEquation, 'testdata/vdp_test.mat', 500, 3,0.05, Generate = False)
    Pendulum            = System(PendulumEquation, 'testdata/pendulum_test.mat', 500, 1, 0.1, Generate = False)
    controlledVanDerPol = System(controlledVanDerPolEquation, 'testdata/controlled_test.mat', 500, 5, 0.05, Generate = False)

if __name__ == "__main__":
    main()


