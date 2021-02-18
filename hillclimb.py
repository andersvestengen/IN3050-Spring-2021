import numpy as np
import matplotlib.pyplot as plt
import itertools as itl
import csv
import time
from scipy import interpolate


with open('european_cities.csv')as f:
  byer = np.asarray(list(csv.reader(f, delimiter=';')))

Sbyer = byer[1:7,0:6].astype(float)

perms = np.asarray(list(itl.permutations('012345')))
perms = perms.astype(int)


def Defproblem(cities, citylist):
    CityList = byer[1:cities+1,0:cities].astype(float)
    perms = np.asarray(list(itl.permutations(range(cities))))
    perms = perms.astype(int)
    return CityList, perms

def Exhaustive(mat, listing):
    fitness = np.zeros(len(listing))
    for i in range(len(listing)):
        fDist = 0
        for k in range(len(listing[0,:])-1):
            #print("Citiesexhauast:", len(listing[0,:])-1)
            fDist += float(mat[listing[i,k],listing[i,k+1]])
        
        fDist += float(mat[listing[i,0],listing[i,-1]])
        fitness[i] = fDist
                       
    key2 = np.where(fitness == np.amin(fitness))
    return fitness[key2[0][0]], listing[key2[0][0]]


def Exhaustive2(mat, listing):
    fitness = np.zeros(len(listing))
    for i in range(len(listing)):
        fDist = 0
        for k in range(len(listing)-1):
            fDist += float(mat[listing[k],listing[k+1]])
        
        fDist += float(mat[listing[0],listing[-1]])
        fitness[i] = fDist
                       
    key2 = np.where(fitness == np.amin(fitness))
    return fitness[key2[0]], listing[key2[0]]

@profile
def Hill_climb(runs, cmat, routeseed):
    #Adapted from Marsland p.205
    cities = np.max(routeseed)
    City_order = routeseed
    distance = 0
    for j in range(cities):
        distance += cmat[City_order[j], City_order[j+1]]
    distance += cmat[City_order[0], City_order[-1]]

    for i in range(runs):
        city1 = np.random.randint(cities)
        city2 = np.random.randint(cities)
        
        if city1 != city2:
            order = City_order
            order = np.where(order == city1,-1,order)
            order = np.where(order == city2,city1,order)
            order = np.where(order == -1,city2,order)
        
            new_dist = 0

            for j in range(cities):
                new_dist += cmat[order[j], order[j+1]]
            new_dist += cmat[order[0],order[-1]]

            if new_dist < distance:
                distance = new_dist
                City_order = order
                
    return City_order, distance
"""
problem_size = 8
Cities, routes = Defproblem(problem_size, byer)
seed = np.random.randint(len(routes))
distancem, route = Exhaustive(Cities, routes)
print("best route is cities:", route, "with total distance:", distancem)

Hroute = np.zeros((20, problem_size))
Hdistance = np.zeros(20)
for i in range(20):
    Cities, routes = Defproblem(problem_size, byer)
    seed = np.random.randint(len(routes))   
    order, distance = Hill_climb(1000, Cities, routes[seed])
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
problem_size = 10
arraysize = 20
Hroute = np.zeros((arraysize, problem_size))
Hdistance = np.zeros(arraysize)

for i in range(arraysize):
    Cities, routes = Defproblem(problem_size, byer)
    seed = np.random.randint(len(routes))   
    order, distance = Hill_climb(200, Cities, routes[seed])
    Hdistance[i] = distance
    Hroute[i] = order
    
plt.plot(np.arange(len(Hdistance)), Hdistance, label='HillAlg')
#plt.axhline(y=distancem, color='r', linestyle='-', label='ExhaustiveBest')
plt.legend()
plt.show()

key3 = np.where(Hdistance == np.amin(Hdistance))
print("Hill climb found order:", Hroute[key3[0][0]], "to be the best, with a distance of:", Hdistance[key3[0][0]])
