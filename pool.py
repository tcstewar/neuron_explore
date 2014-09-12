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

    def get_voltage(self):
        return self.soma.voltage.copy()

    def get_syn_current(self):
        return self.syn.current.copy()


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
    def get_voltage(self):
        return self.soma.voltage.astype(float) / 0x10000
    def get_syn_current(self):
        return self.syn.current.astype(float) / 0x10000

class CompactPool(Pool):
    def __init__(self, n_neurons, dt=0.001, tau_syn=0.008, weight_syn=8.0):
        self.weight_shift = 0
        self.weight_syn = weight_syn
        while (1 << self.weight_shift) < weight_syn:
            self.weight_shift += 1

        self.state = np.zeros(n_neurons, dtype='uint32')
        # bits  5- 0: voltage
        # bits  7- 6: refractory
        # bits 13- 8: bias
        # bits 28-14: current
        # bits 29-31: decay_shift

        decay_shift = 0
        while (1 << decay_shift) * dt < tau_syn:
            decay_shift += 1
        assert 0 < decay_shift < 8
        self.state |= decay_shift << 29

        # just used by testing code
        self.soma = soma.LIFFixedMinimal(n_neurons, dt=dt, tau_rc=0.016,
                                         tau_ref=0.02, bits=6)

    def set_bias(self, bias):
        rc_shift = 4
        bias = (bias * 0x10000).astype('i32') >> (rc_shift + 2)
        bias = bias & 0x0000FC00
        self.state |= bias

    def get_bias(self):
        bias = ((self.state >> 8) & 0x000003F).astype('i16')
        bias[bias > 0x1F] -= 0x40  # handle sign on current
        return bias.astype(float) / 2


    def get_voltage(self):
        voltage = self.state & 0x0000003F
        return voltage.astype(float) / 0x40

    def get_syn_current(self):
        current = ((self.state >> 14) & 0x7FFF).astype('i16')
        current[current > 0x3FFF] -= 0x8000  # handle sign on current
        return current.astype(float) / (1 << 10)

    def step(self, spikes):
        # extract data out of the state
        decay_shift = (self.state & 0xE0000000) >> 29
        voltage = self.state & 0x0000003F
        refractory = (self.state >> 6) & 0x0000003
        bias = ((self.state >> 8) & 0x000003F).astype('i16')
        bias[bias > 0x1F] -= 0x40  # handle sign on current
        current = ((self.state >> 14) & 0x7FFF).astype('i16')
        current[current > 0x3FFF] -= 0x8000  # handle sign on current

        # synaptic decay
        decay = current >> decay_shift
        current -= decay
        current[decay==0] = 0

        # add spike
        current += spikes << (self.weight_shift + (15-4) - decay_shift)

        # soma update
        rc_shift = 4
        dv = (((current >> 5) - voltage) >> rc_shift) + bias

        # no voltage change during refractory period
        dv[refractory > 0] = 0
        refractory[refractory > 0] -= 1

        # update voltage
        voltage = voltage.astype('i8') + dv
        voltage[voltage < 0] = 0

        # detect spikes
        spiked = voltage >= 0x40
        refractory[spiked > 0] = 2
        voltage[spiked > 0] -= 0x40

        # put data back into state
        self.state &= 0xE0003F00
        self.state |= (current << 14) & 0x1FFF8000
        self.state |= (refractory << 6)
        self.state |= voltage

        return spiked



if __name__ == '__main__':
    import pylab
    pools = [CompactPool(5), FixedPool(5), StdPool(5)]
    for i, pool in enumerate(pools):
        pylab.subplot(len(pools), 1, i+1)

        pool.set_bias(np.array([-2, -1, 0, 1, 2]))
        data = []
        for i in range(100):
            D = 5
            pool.step(np.eye(D)[i%D][:5].astype('i32'))
            data.append(pool.get_voltage())

        pylab.plot(data)
    pylab.show()


