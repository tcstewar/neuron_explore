import numpy as np


class Neuron(object):
    pass

class LIF(Neuron):
    def __init__(self, tau_rc=0.02, tau_ref=0.002):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

class Rate(Neuron):
    pass

class Spiking(Neuron):
    pass

class Fixed(Neuron):
    pass

class Izhikevich(Neuron):
    def __init__(self, a=0.02, b=0.2, c=-65, d=8):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

class NeuronPool:
    def __init__(self, n_neurons, neuron_types=None):
        if neuron_types is None:
            neuron_types = self.neuron_types
        for n in neuron_types:
            for key, value in n.__dict__.items():
                if not key.startswith('_'):
                    setattr(self, key, value)
        self.make(n_neurons)

    def make(self, n_neurons):
        raise NotImplementedError('NeuronPools must provide "make"')

    def step(self, dt, J):
        raise NotImplementedError('NeuronPools must provide "step"')



class LIFRatePool(NeuronPool):
    neuron_type = [LIF, Rate]

    def make(self, n_neurons):
        pass

    def step(self, dt, J):
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            r = 1.0 / (self.tau_ref + self.tau_rc * np.log1p(1.0 / (J-1)))
            r[J <= 1] = 0
        finally:
            np.seterr(**old)
        return r * dt   # multiply by dt to do rate per timestep


class LIFSpikingPool(NeuronPool):
    neuron_type = [LIF, Spiking]

    def make(self, n_neurons):
        self.voltage = np.zeros(n_neurons)
        self.refractory_time = np.zeros(n_neurons)

    def step(self, dt, J):
        dv = (dt / self.tau_rc) * (J - self.voltage)
        self.voltage += dv

        self.voltage[self.voltage < 0] = 0

        self.refractory_time -= dt

        self.voltage *= (1-self.refractory_time / dt).clip(0, 1)

        spiked = self.voltage > 1

        overshoot = (self.voltage[spiked > 0] - 1) / dv[spiked > 0]
        spiketime = dt * (1 - overshoot)

        self.voltage[spiked > 0] = 0
        self.refractory_time[spiked > 0] = self.tau_ref + spiketime

        return spiked

class LIFFixedPool(NeuronPool):
    neuron_type = [LIF, Spiking, Fixed]

    def make(self, n_neurons):
        self.voltage = np.zeros(n_neurons, dtype='i64')
        self.refractory_time = np.zeros(n_neurons, dtype='u4')

        self.one_over_tau_rc = int(0x10000 / self.tau_rc)

    def step(self, dt, J):
        J = np.asarray(J * dt * 0x10000, dtype='i64')

        dv = ((J - self.voltage) * self.one_over_tau_rc ) >> 16

        dv[self.refractory_time > 0] = 0

        self.refractory_time[self.refractory_time > 0] -= 1

        self.voltage += dv

        self.voltage[self.voltage < 0] = 0

        spiked = self.voltage > 0x10000

        self.refractory_time[spiked > 0] = int(self.tau_rc / dt)

        self.voltage[spiked > 0] = 0

        return spiked






class IzhikevichPool(NeuronPool):
    neuron_type = [Izhikevich, Spiking]

    def make(self, n_neurons):
        self.v = np.zeros(n_neurons) + self.c
        self.u = self.b * self.v

    def step(self, dt, J):
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + J) * 1000
        du = (self.a * (self.b * self.v - self.u)) * 1000

        self.v += dv * dt
        self.u += du * dt

        spiked = self.v >= 30
        self.v[spiked > 0] = self.c
        self.u[spiked > 0] = self.u[spiked > 0] + self.d

        return spiked


neuron_models = [
    LIFSpikingPool,
    LIFRatePool,
    LIFFixedPool,
    IzhikevichPool,
    ]

import inspect
def create(n_neurons, neuron_type):

    # make sure it's a list
    try:
        len(neuron_type)
    except TypeError:
        neuron_type = [neuron_type]

    # make sure elements in the list are instances, not classes
    for i, type in enumerate(neuron_type):
        if inspect.isclass(type):
            neuron_type[i] = type()

    # look through the list of neuron models to see if we can
    # find a match
    for model in neuron_models:
        for type in neuron_type:
            if type.__class__ not in model.neuron_type:
                break
        else:
            return model(n_neurons, neuron_type)

    raise Exception('Could not find suitable neuron model')


if __name__ == '__main__':

    spiking = create(100, [LIF, Spiking])
    rate = create(100, [LIF, Rate])
    fixed = create(100, [LIF, Fixed])
    iz = create(100, [Izhikevich])
    #iz = create(100, [Izhikevich(a=0.02, b=0.2, c=-50, d=2)])


    J = np.linspace(0, 10, 100)
    dt = 0.001
    T = 1
    spiking_data = []
    rate_data = []
    iz_data = []
    fixed_data = []
    v = []
    for i in range(int(T/dt)):
        spiking_data.append(spiking.step(dt, J))
        rate_data.append(rate.step(dt, J))
        iz_data.append(iz.step(dt, J))
        fixed_data.append(fixed.step(dt, J))
        v.append(iz.v[-1])

    rate_tuning = np.sum(rate_data, axis=0)/T
    spiking_tuning = np.sum(spiking_data, axis=0)/T
    iz_tuning = np.sum(iz_data, axis=0)/T
    fixed_tuning = np.sum(fixed_data, axis=0)/T

    import pylab
    #pylab.subplot(2, 1, 1)
    pylab.plot(J, rate_tuning)
    pylab.plot(J, spiking_tuning)
    pylab.plot(J, iz_tuning)
    pylab.plot(J, fixed_tuning)
    #pylab.subplot(2, 1, 2)
    #pylab.plot(v)
    pylab.show()



