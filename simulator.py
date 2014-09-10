
import nengo
import nengo.config
import nengo.params
import pool
import numpy as np
import collections
import nengo.utils.numpy as npext
from nengo.utils.distributions import Distribution, Uniform

BuiltEnsemble = collections.namedtuple(
        'BuiltEnsemble', ['eval_points', 'encoders', 'intercepts', 'max_rates',
                          'scaled_encoders', 'gain', 'bias'])

class Config(nengo.config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.configures(nengo.Ensemble)
        self[nengo.Ensemble].set_param('fixed', nengo.params.Parameter(False))
        self[nengo.Ensemble].set_param('fixed_bits_soma', nengo.params.Parameter(6))

class Model(object):
    def __init__(self, dt=0.001):
        self.dt = dt
        self.params = {}

class Simulator(object):
    def __init__(self, model, dt=0.001, seed=None, config=Config()):
        self.rng = np.random.RandomState(seed=seed)
        self.pools = {}
        self.model = Model(dt=dt)
        self.dt = dt
        self.config = config

        for ens in model.all_ensembles:
            self.make_pool(ens)

    def find_gain_bias(self, soma, intercepts, max_rates):
        # find gains and biases for given intercepts and max_rates given
        #  an arbitrary tuning curve
        J_max = 10
        max_rate = np.max(max_rates)
        J, rate = soma.tuning_curve(J_min=0, J_max=J_max)
        #while rate[-1] < max_rate:
        #    J_max *= 2
        #    print J_max
        #    J, rate = soma.tuning_curve(J_min=0, J_max=J_max)
        J_threshold = J[np.where(rate <= 0)[0][-1]]

        gains = np.zeros(soma.n_neurons)
        biases = np.zeros(soma.n_neurons)
        for i in range(len(intercepts)):
            index = np.where(rate > max_rates[i])[0]
            if len(index) == 0:
                index = len(rate) - 1
            else:
                index = index[0]
            p = (max_rates[i]-rate[index-1]) / (rate[index] - rate[index-1])
            J_top = p * J[index] + (1-p) * J[index-1]

            gain = (J_threshold - J_top) / (intercepts[i] - 1)
            bias = J_top - gain
            gains[i] = gain
            biases[i] = bias
        return gains, biases

    def make_pool(self, ens):
        if isinstance(ens.encoders, Distribution):
            encoders = ens.encoders.sample(ens.n_neurons, ens.dimensions,
                                           rng=self.rng)
        else:
            encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
            encoders /= npext.norm(encoders, axis=1, keepdims=True)


        if self.config[ens].fixed:
            p = pool.FixedPool(ens.n_neurons,
                               bits_soma=self.config[ens].fixed_bits_soma)
        else:
            p = pool.StdPool(ens.n_neurons)
        intercepts = nengo.builder.sample(ens.intercepts, ens.n_neurons,
                                          rng=self.rng)
        max_rates = nengo.builder.sample(ens.max_rates, ens.n_neurons,
                                          rng=self.rng)
        gain, bias = self.find_gain_bias(p.soma, intercepts, max_rates)
        p.set_bias(bias)

        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

        self.pools[ens] = p

        self.model.params[ens] = BuiltEnsemble(intercepts=intercepts,
                                               max_rates=max_rates,
                                               gain=gain,
                                               bias=bias,
                                               encoders=encoders,
                                               scaled_encoders=scaled_encoders,
                                               eval_points=None,
                                               )


    def compute_tuning_curves(self, ens, T=1):
        assert ens.dimensions == 1
        p = self.pools[ens]
        A = []
        x = np.linspace(-1, 1, 50)
        for xx in x:
            J = np.dot([xx], self.model.params[ens].scaled_encoders.T)# + p.bias
            print xx, J[0], p.bias[0]
            data = np.zeros(ens.n_neurons)
            for i in range(int(T/self.dt)):
                spikes = self.poisson_spikes(J / (p.weight_syn * self.dt))
                data += p.step(spikes)
                #data += p.soma.step(J)
            A.append(data / T)
        return x, A

    def poisson_spikes(self, rate):
        sign = np.where(rate > 0, 1, -1)
        rate = rate * sign
        rate[rate<=0.00001] = 0.00001

        time = np.zeros(rate.shape)
        spikes = np.zeros(rate.shape)
        time = -np.log(self.rng.rand(*rate.shape)) / rate
        index = np.where(time < self.dt)
        spikes[index] += sign[index]
        while len(index[0]) > 0:
            time += -np.log(self.rng.rand(*rate.shape)) / rate
            index = np.where(time < self.dt)
            spikes[index] += sign[index]
        return spikes



    def run(self, t):
        pass


if __name__ == '__main__':
    config = Config()
    model = nengo.Network()
    model.config[nengo.Ensemble].max_rates=Uniform(100, 200)
    with model:
        a = nengo.Ensemble(n_neurons=50, dimensions=1)
        config[a].fixed = True
        config[a].fixed_bits_soma = 6

    sim = Simulator(model, seed=1, config=config)
    sim.run(1)

    import pylab
    pylab.figure()
    X, A = sim.compute_tuning_curves(a)
    pylab.plot(X, A)
    pylab.show()


