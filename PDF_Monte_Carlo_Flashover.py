import scipy.stats
import numpy as np
from scipy.stats import norm
import Monte_Carlo_Flashover as mc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#Constant characterization of the simulation
HRR = [2150, 1160,780]
E = [704000,413000,245000]
num_bins = 50
num_simulations = 2*10**4


All_Tres = []
All_Prob = []
norm_params = []
for i in range(len(HRR)):
	t = E[i]/HRR[i]
	All_Tres.append(mc.monte_carlo_simulation(num_simulations, HRR[i], t))
	All_Prob.append(mc.flashover_prob(num_simulations, All_Tres[i]))

	(mu, sigma) = norm.fit(All_Tres[i])
	norm_params.append((mu,sigma))

for i in range(len(norm_params)):
	j = [1,6,16]
	print(f'CBUF1:{j[i]}')
	print(f'	Mean: {norm_params[i][0]}')
	print(f'	Probability of flashover: {All_Prob[i]}')


# add a 'best fit' line
hist1, bins1 = np.histogram(All_Tres[0], bins = num_bins, density = True)
y1 = norm.pdf(bins1, norm_params[0][0], norm_params[0][0])
l1, = plt.plot(bins1, y1, 'r-', linewidth=2, label = 'CBUF1:1')
plt.fill_between(bins1, y1, where=(bins1 >= 0), color='red', alpha = .4)

hist2, bins2 = np.histogram(All_Tres[1], bins = num_bins, density = True)
y2 = norm.pdf(bins2, norm_params[1][0], norm_params[1][1])
l2, = plt.plot(bins2, y2, 'b--', linewidth=2, label = 'CBUF1:6')
plt.fill_between(bins2, y2, where=(bins2 >= 0), color='blue', alpha = .4)


hist3, bins3 = np.histogram(All_Tres[2], bins = num_bins, density = True)
y3 = norm.pdf(bins3, norm_params[2][0], norm_params[2][1])
l3, = plt.plot(bins3, y3, 'g--', linewidth=2, label = 'CBUF1:16')
plt.fill_between(bins3, y3, where=(bins3 >= 0), color='green', alpha = .4)

plt.vlines(0,0,0.006, color = 'black', )
plt.ylim(0,0.006)


#plot
plt.legend(handles = [l1,l2,l3])
plt.xlabel("Temperature Residual (K)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()



