from queue import PriorityQueue
def recieveInput():
    edges:list[tuple[int,int,int]] = []
    nodeCount = int(input("Input Node Numbers : "))
    startNode = int(input("Input Start Node : ")) - 1
    edgeCount = int(input("Input Edge Numbers : "))
    for i in range(edgeCount) :
        start,end,weight= [int(s) for s in input('Edge {0} : '.format(i)).split()]
        start-=1
        end-=1
        edges.append((start,end,weight))
    return {
        'nodeCount':nodeCount,
        'startNode':startNode,
        'edges':edges
    }
def findAtLeastCost(nodeCount:int,startNode:int,edges:list[tuple[int,int,int]]):
    edgeLen = len(edges)
    pq:PriorityQueue[tuple[int,int,int]] = PriorityQueue()
    edgesAtNode:dict[int,list[tuple[int,int,int]]] = {}
    nodeTraverse:list[bool] = [False for i in range(nodeCount)]
    edgeTraverse:list[bool] = [False for i in edges]
    #Define Value Edge_At_Node
    for i in range(edgeLen):
        start,end,weight = edges[i]        
        if start not in edgesAtNode.keys():
            edgesAtNode[start] = []
        if end not in edgesAtNode.keys():
            edgesAtNode[end] = []
        edgesAtNode[start].append((end,weight,i))
        edgesAtNode[end].append((start,weight,i))
    #Setup
    nodeTraverse[startNode] = True
    for end,weight,id in edgesAtNode[startNode]:
        pq.put((-weight,end,id))
    #Loop Spanning Tree
    while(not pq.empty()):
        _,start,id=pq.get()
        if not nodeTraverse[start]:
            #print('start :',edges[id][0],'end :',edges[id][1],'weight :'  ,edges[id][2])
            for end,weight,id2 in edgesAtNode[start]:
                if not nodeTraverse[end]:
                    pq.put((-weight,end,id2))
            edgeTraverse[id] = True
            nodeTraverse[start] = True
    #Find result
    result = [edges[i][2] for i in range(edgeLen)if not edgeTraverse[i]]
    return max(result) if len(result)>0 else 0
if __name__ == '__main__':
    inp = recieveInput()
    print('result :',findAtLeastCost(inp['nodeCount'],inp['startNode'],inp['edges']))