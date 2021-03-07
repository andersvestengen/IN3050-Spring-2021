import numpy as np
import csv 
import matplotlib.pyplot as plt
import time

with open('european_cities.csv')as f:
  Cities = np.asarray(list(csv.reader(f, delimiter=';')))

Clist = Cities[0]
Cmat = Cities[1:].astype('float')

rng = np.random.default_rng()

def Distance(order, cmat=Cmat):
    dist = 0
    for j in range(int(len((order)))):
      dist += cmat[order[j-1], order[j]]

    return dist



def fitness(Population, dist=0):
  Population = Population.tolist()
  reports = np.zeros((len(Population),1))
    
  for i in range(len(Population)):
    reports[i] = Distance(Population[i])
                
  Scorecard = list(zip(Population,reports.tolist()))
    
  res = list(zip(*sorted(Scorecard, key = lambda x: x[1])))
  Spop = np.asarray(res[0])
  if dist == 1:
    Sdist = np.asarray(res[1])
    return Spop, Sdist
  else:
    return Spop


def GenPop(p_number, city):
  peeps = np.zeros((p_number, city)).astype('int')
  for i in range(p_number):
    peeps[i] = rng.permutation(city)
  return peeps

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

def pmx(a, b):
    # Denne er lånt fra LF til uke 3 IN3050
    half = len(a) // 2
    start = np.random.randint(0, len(a)-half)
    stop = start+half
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

def Selection(Population, cities, fitcut, M_rate):
  Fpop, Fdist = fitness(Population, dist=1)
  P_size = len(Population)
  #print("old population is:", Population)
  fitcut = int(len(Population)*fitcut)
  unfit = int(P_size - fitcut)
  N_pop = np.zeros((P_size, cities)).astype('int')
  N_pop[:fitcut] = Fpop[:fitcut] # best 40% are safe from here
  for i in range(fitcut, P_size):
    #print("Starting tournament")
    Winners = Tournament_selection(Population, cities)
    M_s = rng.random()
    c1 = pmx(Winners[0].tolist(), Winners[1].tolist())
    if M_rate > M_s:
      #print("Child Mutated")
      c1 = Mutate(c1)
    N_pop[i] = c1

  return N_pop, Fdist



def Tournament_selection(Population, cities):
  C_pool = np.zeros((10, cities)).astype('int')
  for i in range(10):
    ran = np.random.randint(int(len(Population)), size=1)
    #print("Numpy chose:", ran)
    C_pool[i] = Population[ran]
  S_pop = fitness(C_pool)
  #print("Selected from tournament:", S_pop[:2])
  return S_pop[:2]




def Generation(SeedPpop, SeedCities, runs, fitcut, M_rate):
  Population = GenPop(SeedPpop,SeedCities)
  Fpop, Fdist = fitness(Population, dist=1)
  Genstats = np.zeros((runs, 4)) # min, max, mean, std
  Champs = np.zeros((runs, SeedCities))
  Champ_dist = np.zeros((runs, 1))
  #print("Genstat stuff:", Fdist[0,0], Fdist[-1,0], np.mean(Fdist[:]), np.std(Fdist[:]))
  Genstats[0] = Fdist[0,0], Fdist[-1,0], np.mean(Fdist[:]), np.std(Fdist[:])
  Champs[0] = Fpop[0]
  Champ_dist[0] = Fdist[0]
  for i in range(runs):
    Population, S_dist = Selection(Population, SeedCities, fitcut, M_rate)
    #print(S_dist[0])
    Genstats[i] = S_dist[0,0], S_dist[-1,0], np.mean(S_dist[:]), np.std(S_dist[:])
    #print("Genstat stuff:", S_dist[0], S_dist[-1,0], np.mean(S_dist[:]), np.std(S_dist[:]))
    Champs[i] = Population[0]
    Champ_dist[i] = S_dist[0]
  return Champs, Champ_dist, Genstats


#Problem defined below this line 

City_number = 6
Population_number = 30


Tourlist = GenPop(Population_number, City_number)
fitcut = 0.4



City_number = 24
Population_number = [100, 200, 400]
fitcut = 0.4
runs = 20
M_rate = 0.5
B_routes = np.zeros((runs, City_number, 3))
B_dist = np.zeros((runs, 3))
Timings = np.zeros(3)
for i in range(3):
  start = time.time()
  Cityroutes, Cdistances, Genstats = Generation(Population_number[i], City_number, runs, fitcut, M_rate)
  stop = time.time()
  Timings[i] = stop - start
  B_routes[:,:,i] = Cityroutes
  B_dist[:,i] = Cdistances[:,0]
  P_best = np.asarray(sorted(list(Genstats[:,0].tolist())))
  P_worst = np.asarray(sorted(list(Genstats[:,1].tolist())))
  P_mean = np.mean(np.asarray(sorted(list(Genstats[:,2].tolist()))))
  P_std = np.std(np.asarray(sorted(list(Genstats[:,3].tolist()))))
  print("Fitness of population ", Population_number[i], "was:", "best:", P_best[0], "worst:", P_worst[-1], "mean:", P_mean, "std:", P_std)
  print(" ")
  print("Best route from pop", Population_number[i], "was route ", Clist[Cityroutes[-1].astype('int')], "With a distance of: ", Cdistances[-1,0], "and runtime of:", Timings[i], "Seconds")
  print(" ")


x = np.arange(runs)
for i in range(3):
  labelstuff = 'population: ' + str(Population_number[i])
  plt.plot(x, B_dist[:,i], label=labelstuff)
plt.legend()
plt.show()