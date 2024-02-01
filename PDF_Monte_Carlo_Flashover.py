import scipy.stats
import numpy as np
from scipy.stats import norm
import Monte_Carlo_Flashover as mc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#Constant characterization of the simulation
#HRR = 2150 #kW = KJ/s
#E = 704000 #kJ 
HRR = 780
E = 245000
#HRR = 1160
#E = 413000
t = E/HRR #sec
num_bins = 50
num_simulations = 2*10**4

ALL_Tres = mc.monte_carlo_simulation(num_simulations, HRR, t)
prob = mc.flashover_prob(num_simulations, ALL_Tres)
# best fit of data
(mu, sigma) = norm.fit(ALL_Tres)
print(f'Mean: {mu}')
print(f'Probability of flashover: {prob}')

# the histogram of the data
n, bins, patches  = plt.hist(ALL_Tres, bins = num_bins, density = True)
# add a 'best fit' line
y = norm.pdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

def gauss(x):
	return scipy.stats.norm.pdf(x, mu, sigma)

prob = scipy.integrate.quad(gauss, 0, np.inf)
print(f'Probability of flashover: {prob}')

#plot
plt.xlabel("Temperature Residual (K)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

#plt.hist(ALL_MQH, bins = 50, density = True)
#plt.xlabel("Temperature Residual (K)")
#plt.ylabel("Probability Density")


