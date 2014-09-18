import numpy as np

class Soma(object):
    def __init__(self, n_neurons, dt):
        self.n_neurons = n_neurons
        self.dt = dt

    def tuning_curve(self, J_min=-2, J_max=10, bias=0, T=1):
        # compute tuning curve by feeding in a fixed current to
        #  each neuron (between J_min and J_max) for T seconds
        #  and returning the firing rate in Hz
        J = np.linspace(J_min, J_max, self.n_neurons) + bias
        data = []
        for i in range(int(T/self.dt)):
            data.append(self.step(J))

        tuning = np.sum(data, axis=0)/T
        return J, tuning

    def step(self, J):
        raise NotImplementedError()

# LIF with no refractory period

class LIFMinimal(Soma):
    def __init__(self, n_neurons, dt=0.001, tau_rc=0.02):
        super(LIFMinimal, self).__init__(n_neurons=n_neurons, dt=dt)
        self.voltage = np.zeros(n_neurons)
        self.tau_rc = tau_rc

    def step(self, J):
        dv = (self.dt / self.tau_rc) * (J - self.voltage)
        self.voltage += dv

        self.voltage[self.voltage < 0] = 0

        spiked = self.voltage > 1

        self.voltage[spiked > 0] -= 1

        return spiked

# the standard neuron model from Nengo 2.0

class LIFStandard(Soma):
    def __init__(self, n_neurons, dt=0.001, tau_rc=0.02, tau_ref=0.002):
        super(LIFStandard, self).__init__(n_neurons=n_neurons, dt=dt)
        self.voltage = np.zeros(n_neurons)
        self.refractory_time = np.zeros(n_neurons)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    def step(self, J):
        dv = (self.dt / self.tau_rc) * (J - self.voltage)
        self.voltage += dv

        self.voltage[self.voltage < 0] = 0

        self.refractory_time -= self.dt

        self.voltage *= (1-self.refractory_time / self.dt).clip(0, 1)

        spiked = self.voltage > 1

        overshoot = (self.voltage[spiked > 0] - 1) / dv[spiked > 0]
        spiketime = self.dt * (1 - overshoot)

        self.voltage[spiked > 0] = 0
        self.refractory_time[spiked > 0] = self.tau_ref + spiketime

        return spiked

# LIF Fixed point implementation
#  based on version for SpiNNaker, but with modifications to change multiplies into shifts

class LIFFixed(Soma):
    def __init__(self, n_neurons, bits=16, dt=0.001,
                                  tau_rc=0.02, tau_ref=0.002):
        super(LIFFixed, self).__init__(n_neurons, dt)

        self.voltage = np.zeros(n_neurons, dtype='i32')

        # compute a mask for the voltage so we only use self.bits number of bits
        self.bits = bits
        if bits >= 16:
            self.voltage_bit_mask = 0xFFFF
        else:
            self.voltage_bit_mask = 0xFFFF - ((1 << (16-bits))-1)

        # determine how much shifting is needed to multiply
        #  by tau_rc/dt
        self.rc_shift = 0
        while (1 << self.rc_shift) * dt < tau_rc:
            self.rc_shift += 1

        # number of time steps for refractory period
        self.ref_steps = int(tau_ref / dt)

        # used for adjusting refractory period to do spike time interpolation
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
        self.voltage &= self.voltage_bit_mask   # limit # of bits stored
        return spiked

# Minimal version of fixed-point model.  Removes the LFSR and interpolation
# by assuming that the input current when the model leaves the refractory period
# is the same as the current when it entered the refractory period (since that
# lets you just subtract 1 from the voltage above threshold to get the starting
# voltage after the final refractory period's time step.
#
# as a result, the entire model is just adds and shifts

class LIFFixedMinimal(Soma):
    def __init__(self, n_neurons, bits=16, dt=0.001,
                                  tau_rc=0.02, tau_ref=0.002):
        super(LIFFixedMinimal, self).__init__(n_neurons, dt)

        self.voltage = np.zeros(n_neurons, dtype='i32')

        # compute a mask for the voltage so we only use self.bits number of bits
        self.bits = bits
        if bits >= 16:
            self.voltage_bit_mask = 0xFFFF
        else:
            self.voltage_bit_mask = 0xFFFF - ((1 << (16-bits))-1)

        # determine how much shifting is needed to multiply
        #  by tau_rc/dt
        self.rc_shift = 0
        while (1 << self.rc_shift) * dt < tau_rc:
            self.rc_shift += 1

        # number of time steps for refractory period
        self.ref_steps = int(tau_ref / dt)

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
        #assert np.sum(self.voltage >= 0x10000) == 0
        self.voltage[self.voltage >= 0x10000] = 0xFFFF
        self.voltage &= self.voltage_bit_mask   # limit # of bits stored
        return spiked


if __name__ == '__main__':
    pool = LIFFixedMinimal(n_neurons=100, tau_rc=0.016, tau_ref=0.002, bits=16)
    J, tuning = pool.tuning_curve()
    import pylab
    pylab.plot(J, tuning)
    pylab.show()

