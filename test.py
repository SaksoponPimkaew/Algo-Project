from index import findAtLeastCost
from scipy.optimize import curve_fit
import timeit
import matplotlib.pyplot as plt
import numpy as np

def worstTestCase(n, iterations=1):
    times = []
    cnt = 1
    for i in n:
        i=int(i)
        inp = [[(j,k,cnt)for k in range(j)]for j in range (1,i-1)]
        inp.append([(0,i-1,-1)])
        inp = np.concatenate(inp) 
        time_taken = timeit.timeit(lambda: findAtLeastCost(i,0,inp), number=iterations)
        cnt+=1
        times.append(time_taken)
    return times
def bestTestCase(n, iterations=1):
    times = []
    for i in n:
        i=int(i)
        time_taken = timeit.timeit(lambda: findAtLeastCost(i,0,[(j-1,j,5) for j in range(1,i) ]), number=iterations)
        times.append(time_taken)
    return times

n_values = np.linspace(2, 202, 50)
# Benchmark the function
execution_times = np.multiply(worstTestCase( n_values,5),1000)

# Define the functions to fit

def Logn(x, a, b):
    return a * np.log2(x) + b

def Linear(x, a, b):
    return a * x + b

def NlogN(x, a, b):
    return a * x * np.log2(x) + b

def Square(x, a, b):
    return a * x*x  + b

def N2logN(x,a,b):
    return a*x*x*np.log2(x) + b

def Cube(x,a,b):
    return a*x*x*x + b

funcs = [Logn,Linear,NlogN,Square,N2logN,Cube]

# Fit curves to the data using curve_fit
predicteds = [func(n_values,*curve_fit(func,n_values,execution_times)[0])  for func in funcs]

# Calculate RMSE values
rmses = [np.sqrt(np.mean(np.power((execution_times - predicted) , 2))) for predicted in predicteds]

# Plot the data and fitted curves
plt.scatter(n_values, execution_times, label='Actual Data')
for i in range(len(funcs)):
    plt.plot(n_values, predicteds[i],label=f'({str(funcs[i].__qualname__)} RMSE={rmses[i]:.2f})')
plt.xlabel('Node Count')
plt.ylabel('Runtime')
plt.title('Fitted Curves for Benchmark Data')
plt.legend()
plt.show()