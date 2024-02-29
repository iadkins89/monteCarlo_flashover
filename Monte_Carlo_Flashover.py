#	THIS SCRIPT CONTAINS FUNCTION DEFINITIONS FOR 
#	A MONTE CARLO SIMULATION USED TO PREDICT THE
#	OCCURENCE OF FLASHOVERS

import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#CONSTANTS
###################################################################################################################
#density_ambient 
p= 1.2 #kg/m^3

#specific_heat_ambient
c_p = .001 #kJ/kg-k

#ambient_temp 
"""
T_formula denotes the ambient tepmperature used in the first parathesized term in the MQH formular. 
	Unit analysis shows this term needs to been in units of K
T denotes the ambient temperature in the denominator of the LHS of the MQH formula. Following Bruns
	paper this term is left in unit C.
"""
T_formula = 293.15 #K
T = 20 #deg C

#gravity 
g = 9.81 #m/s^2

#wall_thickness_values
# Brun's paper says these values should be in mm. Unit analysis shows units should be in m.
wall_thickness_values = [.0127,.0159] #m

#wall conductivity sample values 
# Brun's paper has these values in W/m-k. Unit analysis shows this should be in KW/m-k or cw in J/kg-k
kw_samples = [.00028,.000258,.000276,.000254,.000314,.00028,.0003,.00023,.00032, .00026, .000244,.000229] #(kW/m-k)

#specific heat samples values (kJ/kg-k)
cw_samples = [1,1.089,1.017,.963,.891,1,1,1,1, 1.5,.95,.95]

#wall density sample values (kg/m^3)
#last three guessed values from charts in reference paper
pw_samples = [810,711,752,743,805,730,845,745,870,640,725,690]

#mean density and conductivity
#Brun's paper states that the is correlation between conductivity and density but no clear correlation
#amongst the other parameters (e.g pw and cw or cw and kw). Hence, Brun does not state how specific heat
#was sampled. Since the multivariate normal plot for specific heat and conductivity should a better correlation
#than specific heat and density in Brun's paper, I am sample specific heat from the specific heat/conductivity
#multinormal distribution.
mean_pw_kw = (np.average(pw_samples),np.average(kw_samples))
mean_cw_kw = (np.average(cw_samples), np.average(kw_samples))

#covariance matrix for density and wall conductivity
X1 = np.stack((pw_samples,kw_samples), axis=0)
cov_pw_kw = np.cov(X1)
X2 = np.stack((cw_samples,kw_samples), axis=0)
cov_cw_kw = np.cov(cw_samples, kw_samples)
###################################################################################################################


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
		fx = random.uniform(26.5,34.6)

	return fx


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

		#Flashover temperature rise (deg C)
		#Brun's paper contradicts itself saying the range is between 500-600 C and 500-600 K
		delta_T = random.uniform(500,600)

		#Location parameter
		K = random.uniform(1.63,2.77)
		
		#Ceiling height
		H_c = sample_ceiling_height()

		#Floor area
		A_f = sample_floor_area()

		#Door height
		H_o = random.uniform(1.63, H_c)

		#Door width
		W_o = random.uniform(0.81, 3.24)

		#Wall thickness
		L_w = np.random.choice(wall_thickness_values)

		#Room aspect ratio
		nu = random.uniform(0.5,1)

		# specific heat, and density
		p_w, k_w = np.random.multivariate_normal(mean_pw_kw,cov_pw_kw)
		
		#conductivity 
		c_pw, k_w_ignore = np.random.multivariate_normal(mean_cw_kw,cov_cw_kw)

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


