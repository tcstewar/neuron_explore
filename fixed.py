import numpy as np

class Pool(object):
    def __init__(self, n_neurons, bits):
        self.bits = bits
        if bits == 16:
            self.voltage_bit_mask = 0xFFFF
        else:
            self.voltage_bit_mask = 0xFFFF - ((1 << (16-bits))-1)
        self.n_neurons = n_neurons
        self.voltage = np.zeros(n_neurons, dtype='i32')
    def limit_bits(self):
        # ensures only self.bits bits are stored in self.voltage
        self.voltage &= self.voltage_bit_mask


class LIFFixedRefractoryPool(Pool):
    def __init__(self, n_neurons, bits=16, dt=0.001,
                                  tau_rc=0.02, tau_ref=0.002):
        Pool.__init__(self, n_neurons, bits)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.dt = dt
        self.rc_shift = 0
        while (1 << self.rc_shift) * dt < tau_rc:
            self.rc_shift += 1
        # print 'using effective tau_rc of', dt * (1 << self.rc_shift)
        self.ref_steps = int(tau_ref / dt)
        self.lfsr = 1

        self.refractory_time = np.zeros(n_neurons, dtype='u8')

    def step(self, J):
        J = np.asarray(J * 0x10000, dtype='i32')
        dv = (J - self.voltage) >> self.rc_shift
        dv[self.refractory_time > 0] = 0
        self.refractory_time[self.refractory_time > 0] -= 1
        self.voltage += dv
        self.voltage[self.voltage < 0] = 0
        spiked = self.voltage > 0x10000
        self.refractory_time[spiked > 0] = self.ref_steps

        # randomly adjust the refractory period to account for overshoot
        for i in np.where(spiked > 0)[0]:
            p = ((self.voltage[i] - 0x10000) << 16)
            if self.lfsr * dv[i] < p:
                self.refractory_time[i] -= 1
            self.lfsr = (self.lfsr >> 1) ^ (-(self.lfsr & 0x1) & 0xB400)

        self.voltage[spiked > 0] = 0
        self.limit_bits()
        return spiked

class LIFFixedQuickRefractoryPool(Pool):
    def __init__(self, n_neurons, bits=16, dt=0.001,
                                  tau_rc=0.02, tau_ref=0.002):
        Pool.__init__(self, n_neurons, bits)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.dt = dt

        self.rc_shift = 0
        while (1 << self.rc_shift) * dt < tau_rc:
            self.rc_shift += 1
        # print 'using effective tau_rc of', dt * (1 << self.rc_shift)
        self.ref_steps = int(tau_ref / dt)
        self.n_neurons = n_neurons

        self.voltage = np.zeros(n_neurons, dtype='i32')
        self.refractory_time = np.zeros(n_neurons, dtype='u8')


    def step(self, J):
        J = np.asarray(J * 0x10000, dtype='i32')
        dv = (J - self.voltage) >> self.rc_shift
        dv[self.refractory_time > 0] = 0
        self.refractory_time[self.refractory_time > 0] -= 1
        self.voltage += dv
        self.voltage[self.voltage < 0] = 0
        spiked = self.voltage > 0x10000
        self.refractory_time[spiked > 0] = self.ref_steps

        # assume that the current at the end of the refractory period is the
        #  same as at the start
        self.voltage[spiked > 0] -= 0x10000
        self.limit_bits()
        return spiked


class LIFFixedPool(Pool):
    def __init__(self, n_neurons, bits=16, dt=0.001, tau_rc=0.02):
        Pool.__init__(self, n_neurons, bits)
        self.tau_rc = tau_rc
        self.dt = dt
        self.rc_shift = 0
        while (1 << self.rc_shift) * dt < tau_rc:
            self.rc_shift += 1
        # print 'using effective tau_rc of', dt * (1 << self.rc_shift)
        self.n_neurons = n_neurons

        self.voltage = np.zeros(n_neurons, dtype='i32')


    def step(self, J):
        J = np.asarray(J * 0x10000, dtype='i32')
        dv = (J - self.voltage) >> self.rc_shift
        self.voltage += dv
        self.voltage[self.voltage < 0] = 0
        spiked = self.voltage > 0x10000
        self.voltage -= spiked * 0x10000
        self.limit_bits()
        return spiked



def compute_tuning_curve(pool, J_min=-2, J_max=10, bias=0, T=1):
    J = np.linspace(J_min, J_max, pool.n_neurons) + bias
    data = []
    for i in range(int(T/pool.dt)):
        data.append(pool.step(J))

    tuning = np.sum(data, axis=0)/T
    return J, tuning

if __name__ == '__main__':
    import pylab

    classes = [LIFFixedPool,
               LIFFixedRefractoryPool,
               LIFFixedQuickRefractoryPool]

    for i, cls in enumerate(classes):
        pool = cls(n_neurons=100, bits=6, tau_rc=0.016)
        J, tuning = compute_tuning_curve(pool)

        pylab.subplot(len(classes), 1, i + 1)
        pylab.plot(J, tuning)
        pylab.xlabel('J')
        pylab.ylabel(cls.__name__, fontsize='small', rotation=80)
    pylab.show()



