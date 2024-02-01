#	THIS SCRIPT CONTAINS FUNCTION DEFINITIONS FOR 
#	A MONTE CARLO SIMULATION USED TO PREDICT THE
#	OCCURENCE OF FLASHOVERS

import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Constants
#density_ambient 
p= 1.2 #kg/m^3
#specific_heat_ambient
c_p = .001 #kJ/kg-k
#c_p = 1.05
#ambient_temp 
T_formula = 293.15 #K
T = 20 #deg C
#T = 293.15 #K
#gravity 
g = 9.81 #m/s^2
#wall_thickness_values = [12.7,15.9]  #mm
wall_thickness_values = [.0127,.0159] #m
#wall conductivity sample values (kW/m-k)
#kw_samples = [.00028,.000258,.000276,.000254,.000314,.00028,.0003,.00023,.00032]
kw_samples = [.00028,.000258,.000276,.000254,.000314,.00028,.0003,.00023,.00032, .00026, .000244,.000229]
#kw_samples = [.28,.258,.276,.254,.314,.28,.3,.23,.32]
#specific heat samples values (kJ/kg-k)
#cw_samples = [1,1.089,1.017,.963,.891,1,1,1,1]
cw_samples = [1,1.089,1.017,.963,.891,1,1,1,1, 1.5,.95,.95]
#cw_samples = [.001,.001089,.001017,.000963,.000891,.001,.001,.001,.001, .0015,.00095,.00095]
#wall density sample values (kg/m^3)
#pw_samples = [810,711,752,743,805,730,845,745,870]
#last three guessed values from charts in reference paper
pw_samples = [810,711,752,743,805,730,845,745,870,640,725,690]
#mean density and conductivity
mean_pw_kw = (np.average(pw_samples),np.average(kw_samples))
mean_cw_kw = (np.average(cw_samples), np.average(kw_samples))
#covariance matrix for density and wall conductivity
#X = np.stack((pw_samples,kw_samples), axis=0)
cov_pw_kw = np.cov(pw_samples,kw_samples)
cov_cw_kw = np.cov(cw_samples, kw_samples)


#print(cov)
#print(mean_pw_kw)
"""
rv = multivariate_normal(mean_pw_kw, cov, True)
data = np.dstack((pw_samples, kw_samples))
z = rv.pdf(data)
print(z)
#plt.contourf(pw_samples, kw_samples, z, cmap='coolwarm')
#plt.show()
"""
"""
#Ceiling height pdf
bins = np.array([2.13,2.29,2.44,2.59,3.66])
#freqs = np.array([3.56,41.34,38,30.7])
density= np.array([.19,2.43,2.23,.25])
#widths = bins[1:] - bins[:-1]
#heights = freqs/widths
plt.fill_between(bins.repeat(2)[1:-1], density.repeat(2), facecolor='steelblue')
plt.show()
"""
"""
hist, bin_edges = np.histogram(rel_freq, bins, density=True)
fig, ax = plt.subplots()
ax.bar(bin_edges, hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
plt.show()
"""
"""
data = [3.56,41.34,38,30.7]
#data = np.array([.19,2.43,2.23,.25])
bins = [2.13,2.29,2.44,2.59,3.66]
hist, bin_edges = np.histogram(data, bins=bins, density=False)
print(hist)
print(bin_edges)
plt.hist(hist)
plt.show()
"""

#Function to ensure ceiling height is between min and max range
"""
	bins = np.array([2.13, 2.29, 2.44, 2.59, 3.66])
	freqs = np.array([356, 4134, 380, 307])

	# Normalize the frequencies
	prob_dist = freqs / freqs.sum()

	# Sample a single value from the bin edges based on the specified probabilities
	bin_index = np.random.choice(np.arange(len(bins)-1), size=1, p=prob_dist)[0]
	sampled_value = np.random.uniform(bins[bin_index], bins[bin_index + 1], size=1)[0]

	return sampled_value
"""
def sample_ceiling_height():

	bins = np.array([2.13, 2.29, 2.44, 2.59])
	density = np.array([0.19, 2.43, 2.23, 0.25])
	prob_dist = density / density.sum()
	
	x = np.random.choice(bins, size = 1, p=prob_dist)

	if x == 2.13:
		fx = random.uniform(2.13,2.29)
	if x == 2.29:
		fx = random.uniform(2.29,2.44)
	if x == 2.44:
		fx = random.uniform(2.44,2.59)
	if x == 2.59:
		fx = random.uniform(2.59,3.66)

	return fx

def sample_floor_area():

	bins = [12.5,13,14,15,17.2,19,21,22.5,26.5]
	density = np.array([0.072,0.08,0.13,0.135,0.105,0.075,0.039,0.018,0.005])
	prob_dist = density / density.sum()
	
	x = np.random.choice(bins, size = 1, p = prob_dist)

	if x == 12.5:
		fx = random.uniform(12.5,13)
	if x == 13:
		fx = random.uniform(13,14)
	if x == 14:
		fx = random.uniform(14,15)
	if x == 15:
		fx = random.uniform(15,17.2)
	if x == 17.2:
		fx = random.uniform(17.2,19)
	if x == 19:
		fx = random.uniform(19,21)
	if x == 21:
		fx = random.uniform(21,22.5)
	if x == 22.5:
		fx = random.uniform(22.5,26.5)
	if x == 26.5:
		fx = random.uniform(26.5,35)

	return fx

#Function to ensure floor area is between min and max range
def trim_floor_area(x):

	if x <= 12.8:
		return 12.8

	if x >= 34.6:
		return 34.6

	return x

#Calculate thermal penetration
def calc_thermal_penetration(p_w, c_w, L_w, k_w):
	return (p_w*c_w*(L_w**2))/(4*k_w)

#Calculate heat transfer coefficient
def heat_transfer_coeff(k_w, p_w, c_w, L_w, t, t_w):

	if t <= t_w:
		return np.sqrt((k_w*p_w*c_w)/t)

	else: 
		return k_w/L_w

#Calculate total wall area
def total_wall_area(A_f, H_c, nu, H_o, W_o):
	return 2*A_f + 2*H_c*(1+nu)*np.sqrt(A_f/nu) - (H_o*W_o)


#Monte Carlo simulation for T residual
def monte_carlo_simulation(num_simulations, HRR, t):

	ALL_Tres = []
	printProgressBar(0, num_simulations, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for j in range(num_simulations):

		#Flashover temperature rise (deg K)
		#delta_T = random.uniform(773.15, 873.15)
		#Flashover temperature rise (deg C)
		delta_T = random.uniform(500,600)

		#Location parameter
		K = random.uniform(1.63,2.77)
		
		#Ceiling height (check mean and std later)
		#H_c = np.random.lognormal(2.61, 0.765)
		#H_c = np.random.normal(2.61, 0.765)
		#H_c = trim_ceiling_height(H_c)
		H_c = sample_ceiling_height()
		#H_c = 1.13

		#Floor area (check mean and std later)
		#_f = np.random.lognormal(18.2, 10.9)
		#A_f = np.random.normal(18.2, 10.9)
		#A_f = np.random.lognormal(18.2, 2.26)
		A_f = sample_floor_area()
		#A_f = 12.8

		#Door height
		H_o = random.uniform(1.63, H_c)

		#Door width
		W_o = random.uniform(0.81, 3.24)

		#Wall thickness
		L_w = np.random.choice(wall_thickness_values)
		#Room aspect ratio
		nu = random.uniform(0.5,1)

		"""
		#Wall conductivity (change later)
		#k_w = 0.000269
		k_w = random.uniform(.00022,.00032)
		
		#Wall density (change later)
		#p_w = 757
		p_w = random.uniform(650,875)
		"""

		p_w, k_w = np.random.multivariate_normal(mean_pw_kw,cov_pw_kw)
		#k_w = np.random.normal(np.average(kw_samples),np.std(kw_samples))
		#p_w = np.random.normal(np.average(pw_samples),np.std(pw_samples))
		
		#Wall specific heat capacity (change later)
		#c_pw = .88
		#c_pw = random.uniform(.88,1.5)
		c_pw = np.random.normal(np.average(cw_samples), np.std(cw_samples))
		#c_pw, k_w_ignore = np.random.multivariate_normal(mean_cw_kw,cov_cw_kw)

		#Thermal penetration coefficient
		t_w = calc_thermal_penetration(p_w, c_pw, L_w, k_w)

		#Heat transfer coefficient
		h_w = heat_transfer_coeff(k_w, p_w, c_pw, L_w, t, t_w)

		#Total wall area
		A_w = total_wall_area(A_f, H_c, nu, H_o, W_o)
		
		#MQH Correlation
		MQH = (K*(HRR/(p*c_p*H_o*W_o*np.sqrt(g*H_o)*T_formula))**(2/3)) * (((h_w*A_w)/(p*c_p*H_o*W_o*np.sqrt(g*H_o)))**(-1/3))
		HGL = T*MQH
		T_res = HGL - delta_T
		T_res = T_res + 273.15 #Covert from deg C to K
		ALL_Tres.append(T_res)
		printProgressBar(j + 1, num_simulations, prefix = 'Progress:', suffix = 'Complete', length = 50)
	return ALL_Tres

def flashover_prob(num_simulations, ALL_Tres):
	#return (1/num_simulations)*sum(np.heaviside(ALL_Tres,.5))
	prob = 0
	for res in ALL_Tres:
		if res >= 0:
			prob = prob + (1/num_simulations)*np.heaviside(res,.5)
	return prob

def relative_error(num_simulations, prob):
	return np.abs((prob - (1/np.sqrt(num_simulations))))/(1/np.sqrt(num_simulations))

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
   
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


