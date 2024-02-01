import Monte_Carlo_Flashover as mc8
import matplotlib.pyplot as plt

#Constant characterization of the simulation
HRR = 2150 #kW
E = 704000 #kJ
t = E/HRR #sec

simulations = [10**1, 10**2,10**3,10**4]
probs = []
errors = []

for sim in simulations:

	ALL_Tres = mc.monte_carlo_simulation(sim, HRR, t)
	prob = mc.flashover_prob(sim, ALL_Tres)
	error = mc.relative_error(sim,prob)
	probs.append(prob)
	errors.append(error)

print("Errors:")
print(errors)
print('\n')
print("Probabilities:")
print(probs)

fig, ax = plt.subplots()
scatter = ax.scatter(simulations, errors)
#plt.scatter(simulations,errors)
plt.show()