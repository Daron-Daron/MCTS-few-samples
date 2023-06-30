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
import matplotlib.pyplot as plt




def lowerintegrand(mu,sigma,g,v):
	row_of_samples = samples[g,v,:]
	E = [ ( (x - mu)**2 / sigma**2) for x in row_of_samples]
	C = sigma**(-num_samples)
	return C* np.exp((-1)*sum(E)) 


def upperintegrand(mu,sigma,g,v,new_reward):
	row_of_samples = samples[g,v,:]
	E = [(x - mu)**2 / sigma**2 for x in row_of_samples]+[(new_reward - mu)**2 / sigma**2 ]
	C = sigma**(-num_samples)
	return C* np.exp((-1)*sum(E)) 


def draw(user_group,item):
	mean  = mu[user_group,item]
	variance = sigma[user_group,item]
	x = np.random.normal(mean,variance**(1/2))
	return x

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
	
	
	

	 
def get_artificial_reward(prior, test_item):

	test_group = draw_from_prior(prior)
	
	new_prior = np.zeros(num_groups) 

	for g in range(0,num_groups):
		s = np.random.choice(range(0,num_samples))
		new_prior[g] = prior[g]* update_factors[g,test_item,s]
 				
	new_prior = list(new_prior)

	if test_group == new_prior.index(max(new_prior)):
		return 1
	else:
		return 0
	

def UCB(prior,History):

	N= 3000 #run for this many turns
 


	rewards_per_arm = np.array([0 for n in range(0, num_items)])
	pulls_per_arm = np.array([1 for n in range(0, num_items)])
	maximand = [1 for n in range(0, num_items)] 

	
	print("in UCB History =", History)

	for x in History: 		# This removes the previously pulled arms from consideration.
		rewards_per_arm[x] = -N
		pulls_per_arm[x] =  N


	for t in range(0,N):
		new_UCB_item = np.argmax(maximand)	
		new_UCB_reward = get_artificial_reward(prior, new_UCB_item) 
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
	 
	#print("\n maximand  = ", maximand)
			
	#print("\n pulls per arm  = \n", pulls_per_arm)
	#print("\n rewards per arm  = \n", rewards_per_arm)
	rewards_per_arm = list(rewards_per_arm)
	final_answer = np.argmax(rewards_per_arm)

	print("final answer = ",  final_answer)
	 
	#print("\n maximand  = ", maximand)

	return(final_answer)


 

def new_user(num_turns,correct_group):

	accuracy = []

	print("new user comes from group", correct_group)


	prior = [1 for n in G] 
	History = [] 
	prior = np.array(prior)

	for T in range(0,num_turns):

	
		prior = prior/sum(prior)
 
		print("running UCB. . . ")
		new_item = UCB(prior,History) 

		History = History + [new_item]
		print("in new_user History =", History)

		print("new item = ", new_item)

		new_reward = draw(correct_group, new_item)
		print("new reward = ", round(new_reward,4))

		lower_integrals = list(range(0,num_groups))
		upper_integrals  = list(range(0,num_groups))

		for g in G:
 
			f = lambda y,x: lowerintegrand(x,y,g,new_item)
			#x = integrate.dblquad(f, 0, 5,0,2)[0]
			#print(x)
			lower_integrals[g] = integrate.dblquad(f, 0, 5,0,2)[0] # mean is uniform on [0,5], std uniform on [0,2]

			f = lambda y,x: upperintegrand(x,y,g,new_item,new_reward)
			upper_integrals[g] = integrate.dblquad(f, 0, 5,0,2)[0] 

			prior[g] = prior[g] * upper_integrals[g] 
			prior[g] = prior[g]/ lower_integrals[g]

		prior = prior/sum(prior)
		print("prior = ", [round(x,4) for x in prior])

		
		new_accuracy = round( prior[correct_group],4)
		accuracy = accuracy + [new_accuracy]

	print("accuracy for this user = ", accuracy)
	return(accuracy)

mu = np.loadtxt('mu_goodreads4.csv', delimiter=',')
mu = -mu
print("means = ", mu, "\n") 


sigma = np.loadtxt('sigma_goodreads4.csv', delimiter=',')
print("variances = ",  sigma) 

G = [0,1,2,3]

num_groups =4

num_items = mu.shape[1]

print("there are", num_items, "items")

update_factors = np.load('update_factors_GR_4_clusters_10samples.npy')

print("factors type = ", type(update_factors), "shape = ", update_factors.shape)


num_samples = 10 # Draw this many samples to estimate means and variances

samples = get_samples(mu,sigma,num_samples)
 

Avg_accuracy = np.zeros(20)

num_turns = 20 

for S in range(0,num_turns):
	correct_group = 3
	print("\n \n correct group = ", correct_group)

	X = new_user(20,correct_group)
	plt.plot(X, color = 'black', alpha = 0.3) 

	Avg_accuracy = Avg_accuracy + X

Avg_accuracy = Avg_accuracy/num_turns
  

plt.plot(Avg_accuracy, color = 'black', alpha = 1, linewidth = 5) 


plt.title("New algorithm. 10 data points per group-item. 4 groups. 3,000 UCB turns each. 273 arms.  User from cluster 3. Thick line is average.")
plt.xlabel("# Items Presented")
plt.ylabel("Prior Value on correct Group")

plt.show()
  
	 
