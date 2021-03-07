import numpy as np
import matplotlib.pyplot as plt
import itertools as itl
import csv
import time
from scipy import interpolate
from numpy.random import default_rng
import random
from scipy import stats


with open('european_cities.csv')as f:
  byer = list(csv.reader(f, delimiter=';'))


def Defproblem(cities, nolist = 1):
    #CityList = byer[1:cities+1,0:cities]
    CityList = byer[1:cities+1][0:cities]
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


def GenPop(pop, cities):
    rng = default_rng()
    Population = []
    for i in range(pop):
        candidate = rng.choice(cities, size=cities, replace=False)
        Population.append(candidate)
    return Population

def fitness(cities, Population): # Also used as a tournament selector
    
    reports = np.zeros((len(Population),1))
    #print("This should be array 3", int(len(Population[3])))
    
    for i in range(len(Population)):
            reports[i] = Distance(cities, Population[i])
            
    Scorecard = np.asarray(list(zip(Population,reports)), dtype=object)
    
    res = np.asarray(sorted(Scorecard, key = lambda x: x[1]))
    return res


def pmx(a, b, start, stop):
    child = [None]*len(a)
    
    # Copy a slice from first paret:
    child[start:stop] = a[start:stop]
    
    # Map the same slice in parent b to child using indices from parent a:
    for ind, x in enumerate(b[start:stop]):
        ind += start
        if x not in child:
            while child[ind] != None:
                ind = b.index(a[ind])
            child[ind] = x
    # Copy over the rest from parent b
    for ind, x in enumerate(child):
        if x == None:
            child[ind] = b[ind]
            
    return np.asarray(child)


def pmx_pair(a, b):
    half = len(a) // 2
    start = np.random.randint(0, len(a)-half)
    stop = start+half
    return pmx(a, b, start, stop), pmx(b, a, start, stop)





Cities = 2
popsize = [200, 300, 400]
mutations = 0
runs = 20
fitcuts = 0.2
NCpool = 50

tours = Defproblem(Cities)

print("tours is", tours)

Population = GenPop(10, Cities)

print(Population)