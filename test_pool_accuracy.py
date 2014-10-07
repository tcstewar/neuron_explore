import pool
import numpy as np
import simulator
import nengo

# just used to generate spikes
sim = simulator.Simulator(nengo.Network())

N = 500

pools = [pool.CompactPool(N), pool.StdPool(N)]

bias = np.random.randint(-20, 20, N)

for pool in pools:
    pool.set_bias(bias)

rate = np.random.uniform(-1000, 1000, size=N)
total = np.zeros((len(pools), N))

for i in range(2000):
    spikes = sim.poisson_spikes(rate)

    total[0] += pools[0].step(spikes)
    total[1] += pools[1].step(spikes)


non_zero = np.where(total[0]>0)

rmse_spikes = np.sqrt(np.mean((total[0][non_zero]-total[1][non_zero])**2))
print 'rmse', rmse_spikes / np.mean(total[:,non_zero])
