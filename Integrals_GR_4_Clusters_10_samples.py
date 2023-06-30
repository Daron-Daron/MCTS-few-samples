from scipy.optimize import minimize
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import time
import pandas
from openpyxl import load_workbook
import numpy as np
import csv 



from scipy import integrate
 

def lowerintegrand(mu,sigma,g,v):
	row_of_samples = samples[g,v,:]
	E = [ ( (x - mu)**2 / sigma**2) for x in row_of_samples]
	#print(E)
	#A = (-1)*sum(E)
	C = sigma**(-num_samples)
	return C* np.exp((-1)*sum(E)) 


def upperintegrand(mu,sigma,g,v,i):
	row_of_samples = samples[g,v,:]
	E = [(x - mu)**2 / sigma**2 for x in row_of_samples]+[(row_of_samples[i] - mu)**2 / sigma**2 ]
	#print(E)	
	#A = (-1)*sum(E)
	C = sigma**(-num_samples)
	return C* np.exp((-1)*sum(E)) 
 

def upperintegral(mu,sigma,g,v,n):
	row_of_samples = samples[g,v,:]
	E = [(x - mu)**2 / sigma**2 for x in row_of_samples]+ [(samples[g,v,n]  - mu)**2 / sigma**2] 
	E = -sum(A)
	C = sigma**(-S)
	return C* np.exp(E) 

def get_samples(mu,sigma,S):

	Samples = np.zeros((4,num_items,S)) 
	for n in range(0,4):
		for m in range(0,num_items):
			for i in range(0,S):
				Samples[n,m,i] = np.random.normal(mu[n,m], sigma[n,m]**(1/2))

	return(Samples)
	 

def draw_from_prior(prior):

	total = sum(prior)
	prior = [x/total for x in prior]
	return np.random.choice(G, p=prior)
	
	
	

	 
def get_reward(prior, test_item):


	#for g in G:
		
	#	s = np.choice(G)
	#	prior[g] = prior[g]*upper_integrals[g,test_item,s] / lower_integrals[g,test_item] 

		

	test_group = draw_from_prior(prior)
	
	test_reward = draw(test_group, new_item)	

	prior = [ prior[i]/ sigma[i,new_item] for i in range(0,len(prior))]
	  	
	new_l = [ -(test_reward -mu[g,new_item] )**2 /   ( 2 *(sigma[g, new_item])**2  )     for g in G]
	new_e = np.exp( new_l)
	
	new_prior = list(prior*new_e)

	if test_group == new_prior.index(max(new_prior)):
		return 1
	else:
		return 0
	

def UCB(prior):

	N= 10000 #run for this many turns
 


	rewards_per_arm = np.array([0 for n in range(0, num_items)])
	pulls_per_arm = np.array([1 for n in range(0, num_items)])
	maximand = [1 for n in range(0, num_items)] 

	
	for x in History: 		# This removes the previously pulled arms from consideration.
		rewards_per_arm[x] = -N
		pulls_per_arm[x] =  N


	for t in range(0,N):
		new_UCB_item = np.argmax(maximand)	
		new_UCB_reward = get_reward(prior, new_UCB_item) 
		#print("\n new UCB Reward = ", new_UCB_reward)
		#if new_UCB_reward == 1:
			#print("\n Reward = 1 on turn ", t, "item = ", new_UCB_item)	
		
		#print("\n turn ", t, "item = ", new_UCB_item)
		#print("\n new_UCB_item = ",  new_UCB_item)
		#print("\n pulls per arm  = ", pulls_per_arm)
		#print("\n rewards per arm  = ", rewards_per_arm)
		#print("\n maximand  = ", maximand)
			
		rewards_per_arm[new_UCB_item] =   rewards_per_arm[new_UCB_item] + new_UCB_reward 
		 
		pulls_per_arm[new_UCB_item] =  pulls_per_arm[new_UCB_item] + 1 	
		 

		#print(type(rewards_per_arm),type(pulls_per_arm))
		
		maximand = rewards_per_arm/pulls_per_arm + (0.25* np.log(t+3) / pulls_per_arm)**(1/2) 	
	print("\n new_UCB_item = ",  new_UCB_item)
	print("\n pulls per arm  = ", pulls_per_arm)
	print("\n rewards per arm  = ", rewards_per_arm)
	#print("\n maximand  = ", maximand)
			
	
	rewards_per_arm = list(rewards_per_arm)
	final_answer = np.argmax(rewards_per_arm)

	return(final_answer)


mu = np.loadtxt('mu_goodreads4.csv', delimiter=',')
mu = -mu
print("means = ", mu, "\n") 


sigma = np.loadtxt('sigma_goodreads4.csv', delimiter=',')
print("variances = ",  sigma) 

  
num_items = mu.shape[1]

print("there are", num_items, "items")

num_groups = 4

G = range(0,num_groups)
 
#num_items = mu.shape[1]
 

print("\n there are ", num_items," items \n ")

num_samples = 10 # Draw this many samples to estimate means and variances
 
mu_limit = 5 # uniform prior on mu over [0,5]

sigma_limit =2 #uniform prior on mu over [0,2]

samples = get_samples(mu,sigma,num_samples)
 
Integrals = np.zeros((num_groups,num_items))

for g in range(0,num_groups):
	for v in range(0,num_items):
		f = lambda y, x: (y**(-10) ) *np.exp(   (1/y**2) * samples() )   #x is mean, y is variance
#Integrals = 


#show_samples = [round(x,4) for x in samples[2,num_items,:] ]

#text_group = 0
test_item = 100

#print("g,v = ", text_group , test_item, " \n mean = ", round(mu[text_group,test_item],4), " \n var = ", round(sigma[text_group,test_item],4), "\n vector of samples = ",   show_samples  )

lower_integrals = np.zeros((num_groups,num_items))

for g in range(0,num_groups):
	for v in range(0,num_items):

		f = lambda y,x: lowerintegrand(x,y,g,v)
		lower_integrals[g,v] = integrate.dblquad(f, 0, 5,0,2)[0] #  
		print("\n computed integral", g, " , ", v )

upper_integrals = np.zeros((num_groups,num_items,num_samples))

update_factors = np.zeros((num_groups,num_items,num_samples))

for g in range(0,num_groups):
	for v in range(0,num_items):
		for i in range(0,num_samples):

			f = lambda y,x: upperintegrand(x,y,g,v,i)
			upper_integrals[g,v,i] = integrate.dblquad(f, 0, 5,0,2)[0] # 
			update_factors[g,v,i] = upper_integrals[g,v,i]/ lower_integrals[g,v]
			print("\n computed integral", g, " , ", v, ", ", i )

print(lower_integrals) 
  
  


np.save("lowerintegralsGR4", lower_integrals )
print("saved lower")
np.save("upperintegralsGR4", upper_integrals )
print("saved upper")
np.save("update_factorsGR4", update_factors )
print("saved factors")

