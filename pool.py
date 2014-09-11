import numpy as np

import soma
import synapse

class Pool(object):
    def __init__(self, n_neurons, dt):
        self.n_neurons = n_neurons
        self.bias = 0
        self.dt = dt

class StdPool(Pool):
    def __init__(self, n_neurons, dt=0.001,
                       tau_rc=0.016, tau_ref=0.002,
                       tau_syn=0.008, weight_syn=8.0):
        super(StdPool, self).__init__(n_neurons, dt=dt)

        self.soma = soma.LIFStandard(n_neurons, dt=dt, tau_rc=tau_rc,
                                                tau_ref=tau_ref)
        self.syn = synapse.ExpStandard(n_neurons, dt=dt, tau=tau_syn)
        self.weight_syn = weight_syn
    def set_bias(self, bias):
        self.bias = bias
    def step(self, spikes):
        J = self.syn.step(spikes.astype(float) * self.weight_syn)
        return self.soma.step(J + self.bias)


class FixedPool(Pool):
    def __init__(self, n_neurons, dt=0.001,
                       tau_rc=0.016, tau_ref=0.002,
                       tau_syn=0.008, weight_syn=8.0,
                       bits_soma=6, bits_syn=14):
        super(FixedPool, self).__init__(n_neurons, dt=dt)

        self.soma = soma.LIFFixedMinimal(n_neurons, dt=dt, tau_rc=tau_rc,
                                         tau_ref=tau_ref, bits=bits_soma)
        self.syn = synapse.ExpFixed(n_neurons, dt=dt, tau=tau_syn, bits=bits_syn)
        self.weight_syn = weight_syn
    def set_bias(self, bias):
        self.bias = bias
    def step(self, spikes):
        J = self.syn.step(spikes.astype(float) * self.weight_syn)
        return self.soma.step(J + self.bias)


