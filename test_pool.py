import pool
import numpy as np
import simulator
import nengo
np.random.seed(seed=1)

sim = simulator.Simulator(nengo.Network(), seed=2)

pools = [pool.CompactPool(5), pool.FixedPool(5, bits_syn=15)]

for pool in pools:
    pool.set_bias(np.array([-20, -1, 0, 1, 20]))
    print pool.get_bias()

rate = np.random.uniform(-1000, 1000, size=5)
for i in range(500):
    spikes = sim.poisson_spikes(rate)

    s1 = pools[0].step(spikes)
    s2 = pools[1].step(spikes)

    print 'input', spikes, spikes
    print 'curr ', pools[0].get_syn_current(), pools[1].get_syn_current()
    print 'volt ', pools[0].get_voltage(), pools[1].get_voltage()
    print 'out  ', s1, s2

    assert np.allclose(s1, s2)
    assert np.allclose(pools[0].get_syn_current(), pools[1].get_syn_current())


