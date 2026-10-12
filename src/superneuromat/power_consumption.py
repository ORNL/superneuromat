def estimate_power_comsumption(self, energy_per_spike=23.6, operator_frequency=1.0):

    spike_count = self.ispikes.sum()
    timesteps = len(self.spike_train)

    if timesteps == 0:
        return 0
    else:
        return [(spike_count*energy_per_spike*operator_frequency)/timesteps] * 10**(-3)

