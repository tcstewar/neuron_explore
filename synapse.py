import numpy as np

class Synapse(object):
    def __init__(self, n_synapses, bits=16):
        self.value = np.zeros(n_synapses, dtype='i32')
        self.bits = bits
        if bits == 16:
            self.value_bit_mask = 0xFFFF
        else:
            self.value_bit_mask = 0xFFFF - ((1 << (16-bits))-1)

    def limit_bits(self):
        # ensures only self.bits bits are stored in self.value
        self.value &= self.value_bit_mask

class SynapseStandard(Synapse):
    def __init__(self, n_synapses, bits=16, dt=0.001, tau=0.008):
        Synapse.__init__(self, n_synapses=n_synapses, bits=bits)
        self.dt = dt
        self.decay = np.exp(-dt/tau)
    def step(self, spikes):
        J = np.asarray(spikes << 16, dtype='i32')
        self.value *= self.decay
        self.value += J * (1 - self.decay)
        self.limit_bits()
        return self.value

class SynapseQuick(Synapse):
    def __init__(self, n_synapses, bits=16, dt=0.001, tau=0.008):
        Synapse.__init__(self, n_synapses=n_synapses, bits=bits)
        self.dt = dt
        self.decay_shift = 0
        while (1 << self.decay_shift) * dt < tau:
            self.decay_shift += 1
    def step(self, spikes):
        J = np.asarray(spikes , dtype='i32')

        decay = self.value >> self.decay_shift
        self.value -= decay
        # if we're rounding our decay to 0, jump to 0
        #self.value[decay==0] = 0
        self.value += (J << self.bits)

        shift = self.decay_shift - 16 + self.bits
        if shift < 0:
            return self.value << -shift
        else:
            return self.value >> shift

        return self.value >> (self.decay_shift - 16 + self.bits)


if __name__ == '__main__':
    bits = 16
    tau = 0.128
    syn1 = SynapseStandard(1, tau=tau, bits=bits)
    syn2 = SynapseQuick(1, tau=tau, bits=bits)
    T = 5 * tau
    steps = int(T/syn1.dt)
    input = np.zeros(steps, dtype='i32')
    input[int(10*tau/0.008)] = 1
    input[int(15*tau/0.008)] = 1
    output2 = np.zeros(steps, dtype='i32')
    output1 = np.zeros(steps, dtype='i32')

    for i in range(len(input)):
        output1[i] = syn1.step(input[i])
        output2[i] = syn2.step(input[i])

    import pylab
    pylab.plot(output1, label='sum=%d' % sum(output1))
    pylab.plot(output2, label='sum=%d' % sum(output2))
    pylab.legend()
    pylab.show()


