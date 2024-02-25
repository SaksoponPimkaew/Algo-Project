from index import findAtLeastCost
from scipy.optimize import curve_fit
import timeit
import matplotlib.pyplot as plt
import numpy as np

#------------Big O Func----------------
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

#---------Test Case--------------------

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

def UnitTest():
    pass

def RandomTest():
    pass

def PerformanceTest(maxValue,sampleAmount,testCase):
    nodeCounts = np.linspace(2, maxValue, 50)
    funcs = [Logn,Linear,NlogN,Square,N2logN,Cube]
    execution_times = np.multiply(testCase( nodeCounts,sampleAmount),1000)

    # Fit curves to the data using curve_fit
    predictTime = [func(nodeCounts,*curve_fit(func,nodeCounts,execution_times)[0])  for func in funcs]

    # Calculate RMSE values
    rmses = [np.sqrt(np.mean(np.power((execution_times - predicted) , 2))) for predicted in predictTime]

    # Plot the data and fitted curves
    plt.scatter(nodeCounts, execution_times, label='Actual Data')
    for i in range(len(funcs)):
        plt.plot(nodeCounts, predictTime[i],label=f'({str(funcs[i].__qualname__)} RMSE={rmses[i]:.2f})')
    plt.xlabel('Node Count')
    plt.ylabel('Runtime')
    plt.title('Fitted Curves for Benchmark Data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    PerformanceTest(202,5,bestTestCase)