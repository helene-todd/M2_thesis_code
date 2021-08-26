# M2 Thesis Code

## XPPAUT code

Simulation of a small network of electrically coupled neurons under the leaky integrate-and-fire model.

### 2 Neuron Network

#### if.ode
Two electrically coupled neurons receiving a same input current Iext. The system is not reduced to its dimensionless form.

#### if_dimensionless.ode
Two electrically coupled neurons receiving a same input current Iext.

#### if_dimensionless_pulse.ode
Two electrically coupled neurons receiving a same step input current Iext.

#### phase_reset.ode
Two weakly electrically coupled neurons with a phase difference of 0.5 receiving a same rectangular pulse input current Iext. This causes the phase difference to decrease while keeping the same overall input current, and demonstrates how quickly the system goes from the antiphase stable state in the A/S regime to the stable synchronous state in the A/S regime.

#### keeping_same_phase.ode
Two weakly electrically coupled neurons with a phase difference of 0.5 receiving slightly delayed step currents Iext1 and Iext2. This causes the input current to increase while keeping a phase close to 0.5, and demonstrates how quickly the system goes from the antiphase stable state in the A/S regime to the stable synchronous state in the S regime.

### 5 Neuron Network

#### 5_neurons_phase_reset.ode
Five electrically coupled neurons. Three neurons receive a constant input current I_baseline, while the other two receive a rectangular pulse input current Iext with minimum value I_baseline and maximum value I_pulse. This demonstrates how sychrony of the entire network can be obtained when the input current changes in only two neurons at a very precise timing.

#### 5_neurons_diff_I.ode
Five electrically coupled neurons. Each neuron receives a rectangular pulse input current with slightly different I_baseline values below threshold. This demonstrates how synchrony of the entire network occurs on a specific interval. 

### Bifurcation Diagram

#### bif_weak_coupling.ode
The differential equation for phase difference under weak coupling. 

## Notes

### The 'with_noise' directory
I had to omit a few .dat files due to their large size (~1.4GB). However, these files can easily be generated using XPPAUT since the .ode file and parameters I
 used are included in their respective folders. It can also be worth mentioning that one can slightly modify the source code of XPPAUT (more precisely 'integrate.c') in order to achieve a greater decimal precision in the .dat output file when exporting data (by default it is 6 significant digits, which is problematic for t > 1000 with dt = 0.001).
