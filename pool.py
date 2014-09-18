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
        #bias = (bias * 2).astype('i32')
        #bias = bias.astype(float) / 2
        self.bias = bias
    def get_bias(self):
        return self.bias
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
        self.bias = (bias * 2).astype('i32')
    def get_bias(self):
        return self.bias.astype(float) / 2
    def step(self, spikes):
        J = self.syn.step(spikes.astype(float) * self.weight_syn)
        s = self.soma.step(J + self.get_bias())
        return s
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
        # bits 15- 8: bias
        # bits 31-16: current

        decay_shift = 0
        while (1 << decay_shift) * dt < tau_syn:
            decay_shift += 1
        self.decay_shift = decay_shift

        # just used by testing code
        self.soma = soma.LIFFixedMinimal(n_neurons, dt=dt, tau_rc=0.016,
                                         tau_ref=0.002, bits=6)

    def set_bias(self, bias):
        bias = (bias * 2).astype('i32')
        bias = bias & 0x000000FF
        self.state |= bias << 8

    def get_bias(self):
        bias = ((self.state >> 8) & 0x00000FF).astype('i32')
        bias[bias > 0x7F] -= 0x100  # handle sign on current
        return bias.astype(float) /2.0


    def get_voltage(self):
        voltage = self.state & 0x0000003F
        return voltage.astype(float) / 0x40

    def get_syn_current(self):
        current = ((self.state >> 16) & 0xFFFF).astype('i32')
        current[current > 0x7FFF] -= 0x10000  # handle sign on current
        return current.astype(float) / (1 << 11)

    def step(self, spikes):
        # extract data out of the state
        voltage = (self.state & 0x0000003F).astype('i32')
        refractory = (self.state >> 6) & 0x0000003
        bias = ((self.state >> 8) & 0x00000FF).astype('i32')
        bias[bias > 0x7F] -= 0x100  # handle sign on current
        current = ((self.state >> 16) & 0xFFFF).astype('i32')
        current[current > 0x7FFF] -= 0x10000  # handle sign on current

        current = current << 5
        # synaptic decay
        decay = current >> self.decay_shift
        current -= decay
        current[decay==0] = 0

        current = current >> 5

        # add spike
        current += spikes << (self.weight_shift + 11 - self.decay_shift)

        current[current < -0x8000] = -0x8000
        current[current > 0x7FFF] = 0x7FFF


        total_current = current + (bias << 10)
        # soma update
        rc_shift = 4
        dv = (((total_current >> 5) - voltage) >> rc_shift)

        # no voltage change during refractory period
        dv[refractory > 0] = 0
        refractory[refractory > 0] -= 1

        # update voltage
        voltage = voltage + dv
        voltage[voltage < 0] = 0

        # detect spikes
        spiked = voltage >= 0x40
        refractory[spiked > 0] = 2
        voltage[spiked > 0] -= 0x40
        #voltage[voltage >= 0x40] = 0x3F
        # make sure we're not driving it so hard it spikes twice
        #print voltage >= 0x40
        index = voltage >= 0x40
        '''
        print index
        print self.get_syn_current()[index]
        print current[index]
        print total_current[index]
        print dv[index]
        print self.get_voltage()[index]
        print voltage[index]
        print bias[index]
        print spikes[index]
        '''
        #assert np.sum(voltage >= 0x40) == 0
        voltage[voltage >= 0x40] == 0x3F

        # put data back into state
        self.state &= 0x0000FF00
        self.state |= (current << 16)# & 0xFFFF000
        self.state |= (refractory << 6)
        self.state |= voltage

        return spiked



if __name__ == '__main__':
    import pylab
    pools = [CompactPool(5), FixedPool(5, bits_syn=15), StdPool(5)]
    for i, pool in enumerate(pools):
        pylab.subplot(len(pools), 1, i+1)

        pool.set_bias(np.array([-2, -1, 0, 1, 2]))
        data = []
        for i in range(100):
            D = 5
            pool.step(np.eye(D)[i%D][:5].astype('i32'))
            data.append(pool.get_voltage())
        print data[-5:]

        pylab.plot(data)
    pylab.show()


