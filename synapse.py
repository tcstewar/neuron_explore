import numpy as np

class Synapse(object):
    def __init__(self, n_synapses, dt):
        self.n_synapses = n_synapses
        self.dt = dt
    def step(self):
        raise NotImplementedError()

    def fixed_rate_response(self, rate, weight=1.0, T=1.0):
        if self.n_synapses == 1:
            rates = np.array([rate])
        else:
            rates = np.linspace(0, rate, self.n_synapses)
        result = []
        spike_count = np.zeros(self.n_synapses, dtype='uint32')
        for i in range(int(T/self.dt)):
            input = np.where(spike_count < rates * i * self.dt, 1, 0)
            spike_count += input
            result.append(self.step(input*weight).copy())
        return result, np.mean(result[len(result)/2:], axis=0)

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
            self.current_bit_mask = 0xFFFFFFFF
        else:
            self.current_bit_mask = 0xFFFFFFFF - ((1 << (20-bits))-1)

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
    rate = 1000
    weight = -8.0
    bits = 14
    syn = ExpStandard(n_synapses=50, tau=tau)
    response, avg = syn.fixed_rate_response(rate, weight=weight)
    pylab.figure(1)
    pylab.plot(np.array(response)[:,-1],label='std: %1.3f' % avg[-1])
    pylab.figure(2)
    pylab.plot(avg, label='std')

    syn = ExpFixed(n_synapses=50, tau=tau, bits=bits)
    response, avg = syn.fixed_rate_response(rate, weight=weight)
    pylab.figure(1)
    pylab.plot(np.array(response)[:,-1],label='fixed: %1.3f' % avg[-1])
    pylab.legend()
    pylab.figure(2)
    pylab.plot(avg, label='fixed')
    pylab.legend()


    pylab.show()

