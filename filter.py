import numpy as np

class DecayFilter:
    def __init__(self, tau, dt):
        self.decay = np.exp(-dt/tau)
        self.state = 0
    def step(self, value):
        self.state = self.state * self.decay + value * (1 - self.decay)
        return self.state

class ScaleFilter:
    def __init__(self, tau, dt):
        self.tau = dt / tau
        self.state = 0
    def step(self, value):
        # note that, in fixed point, if tau/dt is a power of 2, these
        # multiplications can all be done as shifts
        self.state = self.state - self.state * self.tau + value * self.tau
        return self.state

T = 1.0          # time to simulate for (s)
dt = 0.001       # timestep size
tau = 0.016      # filter time constant
spikerate = 40   # input rate (Hz)

timesteps = int(T / dt)
t = np.arange(timesteps) * dt

f1 = DecayFilter(tau=tau, dt=dt)
f2 = ScaleFilter(tau=tau, dt=dt)

spikes = np.where(np.random.random(timesteps)<spikerate * dt, 1, 0)

data1 = []
data2 = []
for s in spikes:
    data1.append(f1.step(s))
    data2.append(f2.step(s))



import pylab
pylab.subplot(2, 1, 1)
pylab.plot(t, spikes)
pylab.ylabel('input spikes')
pylab.subplot(2, 1, 2)
pylab.ylabel('filter output')
pylab.plot(t, data1, label='DecayFilter')
pylab.plot(t, data2, label='ScaleFilter')
pylab.legend(loc='lower right')
pylab.show()


