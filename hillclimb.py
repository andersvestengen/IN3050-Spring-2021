import numpy as np
import matplotlib.pyplot as plt
import itertools as itl
import csv
import time
from scipy import interpolate
from numpy.random import default_rng

with open('european_cities.csv')as f:
  byer = np.asarray(list(csv.reader(f, delimiter=';')))

Sbyer = byer[1:7,0:6].astype(float)

perms = np.asarray(list(itl.permutations('012345')))
perms = perms.astype(int)


def Defproblem(cities, nolist = 0):
    CityList = byer[1:cities+1,0:cities].astype(float)
    if nolist == 1:
        return CityList
    else:
        perms = np.asarray(list(itl.permutations(range(cities))))
        perms = perms.astype(int)
        return CityList, perms

def Distance(cmat, order):
    dist = 0
    for j in range(int(len((order)))-1):
        dist += cmat[order[j], order[j+1]]
    dist += cmat[order[0],order[-1]]

    return dist

def Exhaustive(mat, listing):
    fitness = np.zeros(len(listing))
    for i in range(len(listing)):
        fDist = Distance(mat, listing[i])
        fitness[i] = fDist
    key2 = np.where(fitness == np.amin(fitness))
    return fitness[key2[0][0]], listing[key2[0][0]]



def Hill_climb(runs, cmat, routeseed):
    #Adapted from Marsland p.205
    cities = int(len((routeseed)))
    City_order = routeseed
    distance = Distance(cmat, routeseed)

    for i in range(runs):
        city1 = np.random.randint(cities)
        city2 = np.random.randint(cities)
        
        if city1 != city2:
            order = City_order
            order = np.where(order == city1,-1,order)
            order = np.where(order == city2,city1,order)
            order = np.where(order == -1,city2,order)
        
            new_dist = Distance(cmat, order)

            if new_dist < distance:
                distance = new_dist
                City_order = order
                
    return City_order, distance


problem_size = 6
Cities, routes = Defproblem(problem_size)
seed = np.random.randint(len(routes))
distancem, route = Exhaustive(Cities, routes)
print("best route is cities:", route, "with total distance:", distancem)


arraysize = 20
Hroute = np.zeros((arraysize, problem_size))
Hdistance = np.zeros(arraysize)

for i in range(arraysize):
    rng = default_rng()
    seed = rng.choice(problem_size, size=problem_size, replace=False) 
    Cities = Defproblem(problem_size, nolist = 1)
    order, distance = Hill_climb(1000, Cities, seed)  
    Hdistance[i] = distance
    Hroute[i] = order
    
plt.plot(np.arange(len(Hdistance)), Hdistance, label='HillAlg')
plt.axhline(y=distancem, color='r', linestyle='-', label='ExhaustiveBest')
plt.legend()
plt.show()

key3 = np.where(Hdistance == np.amin(Hdistance))
print("Hill climb found order:", Hroute[key3[0][0]], "to be the best, with a distance of:", Hdistance[key3[0][0]])

"""
# All 24 cities
problem_size = 24
arraysize = 20
Hroute = np.zeros((arraysize, problem_size))
Hdistance = np.zeros(arraysize)

for i in range(arraysize):
    rng = default_rng()
    seed = rng.choice(problem_size, size=problem_size, replace=False) 
    Cities = Defproblem(problem_size, nolist = 1)
    order, distance = Hill_climb(1000, Cities, seed)  
    Hdistance[i] = distance
    Hroute[i] = order
    
plt.plot(np.arange(len(Hdistance)), Hdistance, label='HillAlg')
plt.legend()
plt.show()

key3 = np.where(Hdistance == np.amin(Hdistance))
print("Hill climb found order:", Hroute[key3[0][0]], "to be the best, with a distance of:", Hdistance[key3[0][0]])
"""