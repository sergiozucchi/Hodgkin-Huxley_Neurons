import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from types import MethodType

## Constants
C_m=9*np.pi #muF

E_Na=115 #mV
E_K=-12 #mV

V_rest=10.6 #mV

g_Na=1080*np.pi #mS
g_K=324*np.pi #mS
g_m=2.7*np.pi #mS

## Channel intesities
def I_Na(m,h,V):
    return g_Na*(m**3)*(h)*(E_Na-V)

def I_K(n,V):
    return g_K*(n**4)*(E_K-V)

def I_L(V):
    return g_m*(V_rest-V)

## Functions a_x and b_x for the channel's gate
def a_n(V):
    return (10-V)/(100*(np.exp((10-V)/10)-1))

def b_n(V):
    return 0.125*np.exp(-V/80)

def a_m(V):
    return (25-V)/(10*(np.exp((25-V)/10)-1))

def b_m(V):
    return 4*np.exp(-V/18)

def a_h(V):
    return 0.07*np.exp(-V/20)

def b_h(V):
    return 1/(np.exp((30-V)/10)+1)


class HH_Neuron:
    """
    Class for an individual neuron
    """

    def __init__(self,I=280,V0=0,n0=0.4,m0=0.05,h0=0.6,**extra):
        """
        Initialized the object with an external intesity I and initial values {V0,n0,m0,h0}
        """

        # External intensity
        self.I=I

        # Initial values
        self.V=V0
        self.n=n0
        self.m=m0
        self.h=h0

        # Extra attribute for the neuron
        # This will be used for example in the system to indicate if the neuron is excitatory or not
        for key in extra:
            setattr(self, key, extra[key])

    def dV_dt(self):
        """
        Voltage equation
        """
        return (I_Na(self.m,self.h,self.V)+I_K(self.n,self.V)+I_L(self.V)+self.I)/C_m

    def dx_dt(self,x,a_x,b_x):
        """
        Gate activation equation
        """
        return a_x(self.V)*(1-x)-b_x(self.V)*x

    def neuron_ODEs(self,t,y):
        """
        Function that creates an array with the differential equations of the neuron to solve the problem numerically.
        """
        V0,n0,m0,h0=y

        self.V=V0
        self.n=n0
        self.m=m0
        self.h=h0

        return [self.dV_dt(),
                self.dx_dt(self.n,a_n,b_n),
                self.dx_dt(self.m,a_m,b_m),
                self.dx_dt(self.h,a_h,b_h)]

    def run(self,tf,dt,tt=200):
        """
        Integrates the Hodgkin-Huxley equations.

        - Inputs:
        self: HH_Neuron object
        tf: Time of integration
        dt: Step of integration
        tt: Transition time to forget the initial conditions

        - Ouputs:
        sol: Solution output from solve_ivp
        """

        t_eval=np.arange(tt,tt+tf,dt)
        y0=[self.V,self.n,self.m,self.h]

        sol=solve_ivp(self.neuron_ODEs,(0,tf+tt),y0,method="RK45",t_eval=t_eval)

        return sol

    def pulsefreq(self,I,min_peaks=5,tf=1000,dt=0.01):
        """
        Runs and computes the pulsating frequency for the neuron
        """
        self.I=I # Sets the intensity
        res=self.run(tf,dt) # Runs the model

        p,_=find_peaks(res.y[0],height=50) #Finds the peaks

        # Calculates the fequency in [Hz] and the error.
        f=0
        err_f=0
        if len(p)>min_peaks:
            p=p[1:] # Skips the first peak just in case it's not really a peak

            T=np.diff(p)*dt #Calculates the period

            f=1/np.mean(T)
            err_f=(np.std(T)/np.sqrt(len(T)))*f*f

        return f,err_f

class HH_system:

    E_A=60 #mV AMPA volatage
    E_G=-20 #mV GABA volatage

    def f(Vpre):
        """
        Instantaneous function for the neurotransmitter concentration in the presynaptic cleft

        - Vpre is the potential fo the presynaptic neuron
        """
        Tmax=1 #mM^-1
        Vp=63 #mV
        Kp=5 #mV

        return Tmax/(1+np.exp(-(Vpre-Vp)/Kp))

    def dr_dt(self,Vpre,i,a_i,b_i):
        """
        Differential equation for the open fraction of synaptic receptor
        """
        return a_i*HH_system.f(Vpre)*(1-self.r[i])-b_i*self.r[i]

    def I_syn(self,k):
        """
        Defines the I_sync for the neuron k
        """
        I_sync=0

        for i in range(len(self.synp)): # Loop through all synapsis
            d=self.synp[i]
            if d["post"]==k: # Checks when the k neuron is a postsynaptic neuron

                if self.neurons[d["pre"]].isExcitatory: #Boolean to see if it's excitatory (AMPA)

                    I_sync+=d["g"]*self.r[i]*(HH_system.E_A-self.neurons[k].V)

                else: # If it's inhibitory (GABA)

                    I_sync+=d["g"]*self.r[i]*(HH_system.E_G-self.neurons[k].V)

        return I_sync

    def new_dV_dt(self,n):
        """
        New function for the Neuron potential in the system.
        """
        return HH_Neuron.dV_dt(n)+(self.I_syn(n.id))/C_m

    def __init__(self,neurons={},r0=[],synp=[{}]):
        """
        Initializes the system with the neurons, the initial values for r0 and the synapsis.
        The synapsis conections act on the postsynaptic neuron, r0 and synp need to have the same length.

        - neurons is a dictionary of dictionaries with:
        key: The name of the neuron in the system
        value: A dictionary with the attributes for HH_Neuron

        - synp has key values:
        "pre": indicating the key for the presynaptic neuron
        "post": indicating the key for the postsynaptic neuron
        "g": Value of synaptic strength
        """

        self.neurons=dict([(i, HH_Neuron(**neurons[i])) for i in neurons]) # Creates a dictionary with the neurons and the key name
        self.r=r0

        self.synp=synp

        for k,n in self.neurons.items(): # Redefines the differential equation for the potential
            n.id=k
            n.dV_dt = MethodType(self.new_dV_dt,n)

    def system_ODEs(self,t,y):
        """
        Function that creates an array with the differential equations of the neuron to solve the problem numerically.
        """

        # Initializes the array and an indicator
        ODEs=[]
        c=0

        for neur in self.neurons.values(): # Appends the equation for each neuron {V,n,m,h}
            ODEs.append(neur.neuron_ODEs(t,y[c*4:(c+1)*4]))
            c+=1

        for i in range(len(self.synp)):
            # Appends the equation for each neuron {V,n,m,h}
            self.r[i]=y[c*4+i] #Initial condition for r in the synapsis

            d=self.synp[i] # Gets the synapsis dictionary
            npre=self.neurons[d["pre"]] # Gets the presynaptic neuron

            if npre.isExcitatory: #Boolean to see if it's excitatory (AMPA)
                ODEs.append([self.dr_dt(npre.V,i,a_i=1.1,b_i=0.19)])
            else:  # if not, it's inhibitory (GABA)
                ODEs.append([self.dr_dt(npre.V,i,a_i=5,b_i=0.3)])

        return [item for sublist in ODEs for item in sublist] # Returns the equations in a 1D array

    def run_sys(self,tf,dt,tt=400):
        """
        Integrates the system equations.

        - Inputs:
        self: HH_Neuron object
        tf: Time of integration
        dt: Step of integration
        tt: Transition time to forget the initial conditions

        - Ouputs:
        sol: Solution output from solve_ivp
        """

        t_eval=np.arange(tt,tt+tf,dt)
        y0=[item for neur in self.neurons.values() for item in [neur.V,neur.n,neur.m,neur.h]]+self.r

        sol=solve_ivp(self.system_ODEs,(0,tf+tt),y0,method="RK45",t_eval=t_eval)

        return sol
