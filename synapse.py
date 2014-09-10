import numpy as np

class Synapse(object):
    def __init__(self, n_synapses, dt):
        self.n_synapses = n_synapses
        self.dt = dt
    def step(self):
        raise NotImplementedError()

    def fixed_rate_response(self, rate, weight=1.0, T=1.0):
        result = []
        spike_count = 0
        for i in range(int(T/self.dt)):
            if spike_count < rate * i * self.dt:
                spike_count += 1
                input = weight
            else:
                input = 0
            result.append(list(self.step(input)))
        return result, np.mean(result[len(result)/2:])

class ExpStandard(Synapse):
    def __init__(self, n_synapses, dt=0.001, tau=0.016):
        super(ExpStandard, self).__init__(n_synapses=n_synapses, dt=dt)
        self.current = np.zeros(n_synapses, dtype=float)
        self.decay = np.exp(-dt/tau)
    def step(self, spikes):
        self.current *= self.decay
        self.current += (1 - self.decay) * spikes
        return self.current

class ExpFixed(Synapse):
    def __init__(self, n_synapses, dt=0.001, tau=0.016, bits=20):
        super(ExpFixed, self).__init__(n_synapses=n_synapses, dt=dt)
        self.current = np.zeros(n_synapses, dtype='i32')
        self.decay_shift = 0
        while (1 << self.decay_shift) * dt < tau:
            self.decay_shift += 1

        self.bits = bits
        if bits == 20:
            self.current_bit_mask = 0xFFFFF
        else:
            self.current_bit_mask = 0xFFFFF - ((1 << (20-bits))-1)

    def step(self, spikes):
        J = np.asarray((spikes * 0x10000), dtype='i32') >> self.decay_shift

        decay = self.current >> self.decay_shift
        self.current -= decay
        # if we're rounding our decay to 0, jump to 0
        self.current[decay==0] = 0
        self.current += J

        self.current &= self.current_bit_mask

        return self.current.astype('float') / 0x10000



if __name__ == '__main__':
    import pylab
    tau = 0.016
    rate = 300
    weight = 8.0
    syn = ExpStandard(n_synapses=1, tau=tau)
    response, avg = syn.fixed_rate_response(rate, weight=weight)
    pylab.plot(response,label='std: %1.3f' % avg)

    syn = ExpFixed(n_synapses=1, tau=tau, bits=10)
    response, avg = syn.fixed_rate_response(rate, weight=weight)
    pylab.plot(response,label='fixed: %1.3f' % avg)

    pylab.legend()
    pylab.show()
    1/0




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


