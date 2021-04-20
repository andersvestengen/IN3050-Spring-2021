#This section can be run independantly
import numpy as np
import matplotlib.pyplot as plt
import itertools as itl
import csv
import time
from scipy import interpolate
from numpy.random import default_rng
import random
from scipy import stats
rng = default_rng()

with open('european_cities.csv')as f:
  byer = np.asarray(list(csv.reader(f, delimiter=';')))

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

def GenPop(pop, cities):
    rng = default_rng()
    Population = np.zeros((pop,cities))
    for i in range(pop):
        candidate = rng.choice(cities, size=cities, replace=False)
        Population[i,:] = candidate
    return Population.astype(int)


def fitness(cities, Population): # Also used as a tournament selector
    
    reports = np.zeros((len(Population),1))
    #print("This should be array 3", int(len(Population[3])))
    
    for i in range(len(Population)):
            reports[i] = Distance(cities, Population[i])
            
    Scorecard = np.asarray(list(zip(Population,reports)), dtype=object)
    
    res = np.asarray(sorted(Scorecard, key = lambda x: x[1]))
    return res



def crossover43(a, b): # 1 order crossover
    """
    Adapted from: https://www.uio.no/studier/emner/matnat/ifi/INF3490/h16/exercises/inf3490-sol2.pd
    """
    low = high = 0
    while low == high:
        low = int(np.random.randint(1,len(a),1))
        high = int(np.random.randint(1,len(a)+1,1))
    if low > high:
        j = high
        high = low
        low = high

    c = [None]*len(a)
    c[low:high] = a[low:high]

    e1 = e2 = high
    f1 = len(a)
    while None in c:
        if b[e1 % f1] not in c:
            c[e2 % f1] = b[e1 % f1]
            e2 += 1
        e1 += 1
    return np.asarray(c)


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




def Mutate(a): 
    """
    Simple swap mutator
    """
    c1 = 0
    c2 = 0
    while c1 == c2:
        c1 = np.random.randint(len(a)-1)
        c2 = np.random.randint(len(a)-1)
    b = a.copy()
    a[c2] = b[c1]
    a[c1] = b[c2]
    return a


def Mutator(Population, mutation):
    """
    implements crossover and mutations across the selected population. 
    
    """
    newPop = np.zeros((len(Population), len(Population[0])))
    for i in range(len(Population)-1):
        #print("pre crossover parents:", Population[i], Population[i+1])
        #c1 = crossover43(Population[i],Population[i+1])
        #c2 = crossover43(Population[i+1],Population[i])
        c1, c2  = pmx_pair(list(Population[i]), list(Population[i+1]))
        #print("post crossover parents:", c1, c2)
        m = random.uniform(0, 1)
        if m < mutation:
            c1 = Mutate(c1)
            c2 = Mutate(c2)
            #print("after mutation:", c1, c2)
        newPop[i] = c1.astype('int')
        newPop[i+1] = c2.astype('int')
        #print(c1,c2)
    return newPop


def Selection(Population, mutation, NCpool, fitcut, tour):
    newPop = np.zeros((len(Population), len(Population[0])))
    mPop = np.zeros((len(Population), len(Population[0])))
    Spop = []
    for i in range(NCpool):
            rints = rng.integers(low=0, high=fitcut, size=10)
            SeedPop = Population[rints]
            Best = fitness(tour, SeedPop)
            c1,c2 = pmx_pair(list(Best[0,0]), list(Best[1,0]))
            Spop.append(c1)
            Spop.append(c2)
    Spop = np.asarray(Spop)
    print("orig pop as list: ", list(Population))
    print(Population[fitcut:].shape, Spop.shape)
    Tpop = np.concatenate((Population[fitcut:], Spop), axis=0)
    newPop = fitness(tour, Tpop)
    newPop = newPop[:,0]
    cutpoint = int(len(Population) - len(Population[:fitcut]))
    print("This should be array 3", newPop[3])
    newPop = np.concatenate((newPop[:cutpoint], Population[:fitcut,0]), axis=0)
    print(newPop.shape)
    
    return newPop

            

def Generation(problemsize, population, runs, mutation, fitcut, NCpool):
    # Should loop and characterize an entire generation-span
    # fit return     fitpop = res[:,0]
    fitcut = int(population*fitcut)
    tour = Defproblem(problemsize, nolist=1)
    Population = GenPop(population, len(tour))
    #print(Population)
    Genstats = np.zeros((runs, 4)) # min, max, mean, std
    Champs = np.zeros((runs, problemsize))
    Champ_dist = np.zeros(runs)
    n = 10
    for i in range(runs):
        Results = fitness(tour, Population)
        Genstats[i] = Results[0,1], Results[-1,1], np.mean(Results[:,1]), np.std(Results[:,1])
        Champs[i] = Results[0,0]
        Champ_dist[i] = Results[0,1]
        Population = Selection(Population, mutation, NCpool, fitcut, tour)
        #Population = Results[:fitcut,0]
        #Population = Mutator(Population, mutation)
        #SeedPop = GenPop(population, len(tour))
        #print(Population.shape, SeedPop.shape)
        #for j in range(len(Population)):
        #    SeedPop[j] = Population[j]
        #Population = SeedPop
        if i % n == 0:
            print("Gen:", i, "min:", Genstats[i,0], "max:", Genstats[i,1], "mean:", Genstats[i,2], "std:", Genstats[i,3])
    key2 = np.where(Champ_dist == np.amin(Champ_dist))
    key2[0][0]
    print("Champ Champ:", key2[0][0], "min:", Genstats[key2[0][0],0], "max:", Genstats[key2[0][0],1], "mean:", Genstats[key2[0][0],2], "std:", Genstats[key2[0][0],3])
    return Champ_dist
        

                         

Cities = 10
popsize = [200, 300, 400]
mutations = 0
runs = 20
fitcuts = 0.2
NCpool = 50
#champs, dist = Generation(Cities, popsize, runs, mutations, fitcuts)
distances = np.zeros((runs, 3))
for i in range(3):
    distances[:, i] = Generation(Cities, popsize[i], runs, mutations, fitcuts, NCpool)
xline = np.arange(runs)

plt.plot(xline, distances[:,0], label='population1')
plt.plot(xline, distances[:,1], label='population2')
plt.plot(xline, distances[:,2], label='population3')
plt.legend()
plt.show()