import unittest
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

class TestFindAtLeastCost(unittest.TestCase):

    def test_empty_graph(self):
        node_count = 0
        start_node = 0
        edges = []
        result = findAtLeastCost(node_count, start_node, edges)
        self.assertEqual(result, 0)

    def test_single_node_graph(self):
        node_count = 1
        start_node = 0
        edges = []
        result = findAtLeastCost(node_count, start_node, edges)
        self.assertEqual(result, 0)

    def test_fully_connected_graph(self):
        node_count = 3
        start_node = 0
        edges = [
            (0, 1, 1),
            (0, 2, 2),
            (1, 2, 3)
        ]
        result = findAtLeastCost(node_count, start_node, edges)
        self.assertEqual(result, 2)  # Maximum non-traversed edge weight

    def test_disconnected_graph(self):
        node_count = 3
        start_node = 0
        edges = [
            (0, 1, 1),
            (2, 2, 3)  # Isolated node
        ]
        with self.assertRaises(Exception):  # Expect an error (e.g., disconnected graph)
            findAtLeastCost(node_count, start_node, edges)

    def test_invalid_node_count(self):
        node_count = -1
        start_node = 0
        edges = []
        with self.assertRaises(ValueError):  # Expect an error
            findAtLeastCost(node_count, start_node, edges)

    def test_zero_node_count(self):
        node_count = 0
        start_node = 0
        edges = [(1, 2, 1)]  # Invalid edge (node does not exist)
        with self.assertRaises(ValueError):  # Expect an error
            findAtLeastCost(node_count, start_node, edges)

    def test_invalid_edge_count(self):
        node_count = 2
        start_node = 0
        edges = [(0, 1, 1), (0, 1, 2)]  # Duplicate edge
        with self.assertRaises(ValueError):  # Expect an error
            findAtLeastCost(node_count, start_node, edges)

    def test_zero_weight_edge(self):
        node_count = 3
        start_node = 0
        edges = [
            (0, 1, 0),  # Zero weight edge
            (0, 2, 1),
            (1, 2, 3)
        ]
        with self.assertRaises(ValueError):  # Expect an error
            findAtLeastCost(node_count, start_node, edges)

    def test_negative_weight_edge(self):
        node_count = 3
        start_node = 0
        edges = [
            (0, 1, -1),  # Negative weight edge
            (0, 2, 1),
            (1, 2, 3)
        ]
        with self.assertRaises(ValueError):  # Expect an error
            findAtLeastCost(node_count, start_node, edges)

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
