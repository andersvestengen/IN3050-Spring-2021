import numpy as np
import csv 
from itertools import permutations


with open('european_cities.csv')as f:
  Cities = np.asarray(list(csv.reader(f, delimiter=';')))

Clist = Cities[0]
Cmat = Cities[1:].astype('float')



def Distance(order, cmat=Cmat):
    dist = 0
    for j in range(int(len((order)))):
      dist += cmat[order[j-1], order[j]]

    return dist



def fitness(Population):
  Population = Population.tolist()
  reports = np.zeros((len(Population),1))
    
  for i in range(len(Population)):
    reports[i] = Distance(Population[i])
                
  Scorecard = list(zip(Population,reports))
    
  res = list(zip(*sorted(Scorecard, key = lambda x: x[1])))
  Spop = np.asarray(res[0])
  return Spop


def GenPop(p_number, city):
  peeps = np.zeros((p_number, city)).astype('int')
  for i in range(p_number):
    peeps[i] = next(tour)
  return peeps

def pmx(a, b):
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
"""
Anbefalingen er at:

lag en tom array med Population*n_byer

sett newPop = np.zeros((Population, cities))
Sett newPop[]
new_Pop[: int(len(new_Pop)*0.4)] = elites



Extraction theorycrafting: 


N = 40
CS = 24
tour = permutations(range(CS))
Tourlist = np.zeros((N, CS)).astype('int')

for i in range(N):
  Tourlist[i,:] = next(tour)


TcityNames = Cities[0,Tcity]
Rtour = np.random.randint(N,size=1)
print(Distance(Tourlist[Rtour][0]))

"""


#Problem defined below this line 

City_number = 6
Population_number = 30

tour = permutations(range(City_number))

Tourlist = GenPop(Population_number, City_number)
fitcut = 0.4

# ----------------------------------------------------------------------------------------
# Developing selection below this line here

def Selection(Population, cities, fitcut):
  Fpop = fitness(Population)
  P_size = len(Population)
  print("old population is:", Population)
  fitcut = int(len(Population)*fitcut)
  unfit = int(P_size - fitcut)
  N_pop = np.zeros((P_size, cities))
  N_pop[:fitcut] = Fpop[:fitcut] # best 40% are safe from here
  for i in range(fitcut, P_size):
    #print("Starting tournament")
    Winners = Tournament_selection(Population, cities)
    c1 = pmx(Winners[0].tolist(), Winners[1].tolist())
    N_pop[i] = c1
  print("New populatio is:", N_pop)



def Tournament_selection(Population, cities):
  C_pool = np.zeros((10, cities)).astype('int')
  for i in range(10):
    ran = np.random.randint(int(len(Population)), size=1)
    #print("Numpy chose:", ran)
    C_pool[i] = Population[ran]
  S_pop = fitness(C_pool)
  #print("Selected from tournament:", S_pop[:2])
  return S_pop[:2]


Selection(Tourlist, City_number, fitcut)