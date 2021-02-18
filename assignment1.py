# Implement the algorithm here
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
            fDist += float(mat[listing[i,k],listing[i,k+1]])
        
        fDist += float(mat[listing[i,0],listing[i,-1]])
        fitness[i] = fDist
                       
    key2 = np.where(fitness == np.amin(fitness))
    return fitness, fitness[key2[0][0]], listing[key2[0][0]]

start = time.time()
Cities, routes = Defproblem(6, byer)
fitness, distance, route = Exhaustive(Cities, routes)
end = time.time()
elapsed = end - start
print("best route is cities:", route, "with total distance:", distance, "With time spent:", elapsed, "seconds")




timespent = np.zeros(12)
for i in range(1, int(len(timespent))):
    Cities, routes  = Defproblem(i, byer)
    start = time.time()
    Exhaustive(Cities, routes)
    end = time.time()
    timespent[i] = end - start


timespent = timespent[1:]
plt.plot(np.arange(len(timespent)), timespent)
plt.xlabel('cities')
plt.ylabel('time [s]')
plt.show()
for x in timespent:
    print ("{0:.5f}".format(x), "seconds")




x = np.arange(len(timespent))
y = timespent
f = interpolate.interp1d(x, y, fill_value = "extrapolate")
print(f(24), "seconds")   