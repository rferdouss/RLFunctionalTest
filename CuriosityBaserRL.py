# Implementation of curiosity based RL
import copy
import csv
import pickle
import random
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

#Global variables

goalState = 34  # identify the destination room
initialState =0  # identify starting room (initial state of the agent)
maxreward =1000  # max reward for reaching the destination room
maxStepPerEpisode =300
numOfFirehazard =20

flagSparseOrRandom = False
flagRandom = False

epsilonGreedyAlgo = True
numTrainingEpisodes = 1000

statevisitthreshold =4
StateVisitPENALTY =10
FullHealthPoint =100
HealthPointThreshold = 80  # health point threshold, agent will get penalty for loosing health point below this threshold
danzerVal = 5  # loss of health point due to interaction with a hirehazard
CurrentHealthPoint =100
FlagAgentDied = False
#------------------------get and set methods for global variable----------------------------------------------------------------------------------
def getCurrentHealthStatus():
    return CurrentHealthPoint

def setCurrentHealthStatus(val):
    global CurrentHealthPoint
    CurrentHealthPoint = val

def getAgentDeadFlag():
    return FlagAgentDied

def setAgentDeadFlag(val):
    global FlagAgentDied
    FlagAgentDied = val
# ----------------------------------------------------------------------------------------------------------
def GenerateGraph(n,p, outputfilename):
    G = nx.gnp_random_graph(n,p, seed=None, directed=True)
    print('Number of edges ', G.number_of_edges())
    # store graph in a pickle file
    f = open(outputfilename + '.pickle', 'wb')
    pickle.dump(G, f)
    f.close()
    # store the room position where the fire hazard is located
    firehazardpos=[]
    for i in range(numOfFirehazard):
        roomnum = np.random.randint(1, (G.number_of_nodes()+1))
        firehazardpos.append(roomnum)
    f1 = open('results/firehazardposition.pickle', 'wb')
    pickle.dump(firehazardpos, f1)
    f1.close()
    # compute all shortest paths
    #allShortestPaths = {}
    #allShortestPaths = getAllPairShortestPaths(G)
    #f2 = open('results/allshortestpaths.pickle', 'wb')
    #pickle.dump(allShortestPaths, f2)
    #f2.close()
#----------------------------------------------------------------------------------------------------------
def Firehazardlocation(numberofnode):
    # store the room position where the fire hazard is located
    firehazardpos = []
    i = 0
    while i < numOfFirehazard:
        roomnum = np.random.randint(1, (numberofnode+1))
        if roomnum not in firehazardpos:
            # print('pos', pos)
            firehazardpos.append(roomnum)
            i = i + 1
    f1 = open('results/firehazardposition.pickle', 'wb')
    pickle.dump(firehazardpos, f1)
    f1.close()
# ----------------------------------------------------------------------------------------------------------
def loadGraph(graphoutputpath):
    G={}
    G = nx.read_graphml(graphoutputpath)

    # adding doors, buttons, firehazard in rooms
    for n in G.nodes:
        G.nodes[n]['door'] = {g: 0 for g in G.edges(n)}#{g: 0 for g in G.out_edges(n)}  # 0 means door is closed
        #G.nodes[n]['button'] = {g: 0 for g in G.edges(n)}#{g: 0 for g in G.out_edges(n)}   # 0 means button is not pressed
        G.nodes[n]['name'] = 'room'+str(n)
        G.nodes[n]['firehaza    rd'] = 0  # initially no firehazard

    # assigning fire hazard in  rooms
    firehz = pickle.load(open('results/firehazardposition.pickle', 'rb'))
    print('firehazard location = ', firehz)
    for i in range(len(firehz)):
        #roomnum = np.random.randint(0, G.number_of_nodes())
        G.nodes[str(firehz[i])]['firehazard'] = 1  # 1 means firehazard exists inside that room

    #print('Connected = ',nx.is_strongly_connected(G))
    #comp = nx.strongly_connected_components(G)
    #print('strongly connected component ', nx.number_strongly_connected_components(G))
    return G

# ----------------------------------------------------------------------------------------------------------
def Get_RL_environment():
    G = nx.gnp_random_graph(1000, 0.01, seed=None, directed=True)
    return G

# ----------------------------------------------------------------------------------------------------------
def Generate_FireHazard_environment():
    # represent a level of LabRecruits game as a graph.
    G = nx.DiGraph()
    # representing a room as a node in the graph
    G.add_node(0)
    G.add_node(1)
    G.add_node(2) # final destination
    G.add_node(3)
    G.add_node(4)
    G.add_node(5)
    G.add_node(6)
    G.add_node(7)

    #representing the doors as edges (bidirectional means connection from both ways)
    G.add_edges_from([(0, 1), (0, 3)])
    G.add_edges_from([(1, 0), (1, 7)])
    G.add_edges_from([(7, 1), (7, 2)])
    G.add_edges_from([(2, 6),(2, 7)])
    G.add_edges_from([(3, 0), (3, 4), (3, 5)])
    G.add_edges_from([(4, 3), (4, 5)])
    G.add_edges_from([(5, 3), (5, 4), (5, 6)])
    G.add_edges_from([(6, 2), (6, 5)])

    for n in G.nodes:
        G.nodes[n]['door'] = {g: 0 for g in G.out_edges(n)}  # 0 means door is closed
        G.nodes[n]['button'] = {g: 0 for g in G.out_edges(n)}  # 0 means button is not pressed
        G.nodes[n]['name'] = 'room' + str(n)
        #G.nodes[n]['firehazard'] = 0  # initially no firehazard

    # adding danger level as the number of firehazard present in the path (edge) from one room to another (so we consider
    # firehazard to be presented in the path from one room to another that means an edge should contain this info)
    G.edges[(0, 1)]['firehazard'] =7;
    G.edges[(0, 3)]['firehazard'] = 0;
    G.edges[(1, 0)]['firehazard'] = 7;
    G.edges[(1, 7)]['firehazard'] = 7;
    G.edges[(7, 1)]['firehazard'] = 7;
    G.edges[(2, 7)]['firehazard'] = 5;
    G.edges[(7, 2)]['firehazard'] = 5;
    G.edges[(2, 6)]['firehazard'] = 1;
    G.edges[(3, 0)]['firehazard'] = 0;
    G.edges[(3, 4)]['firehazard'] = 1;
    G.edges[(3, 5)]['firehazard'] = 1;
    G.edges[(4, 3)]['firehazard'] = 1;
    G.edges[(4, 5)]['firehazard'] = 1;
    G.edges[(5, 3)]['firehazard'] = 1;
    G.edges[(5, 4)]['firehazard'] = 1;
    G.edges[(5, 6)]['firehazard'] = 1;
    G.edges[(6, 5)]['firehazard'] = 1;
    G.edges[(6, 2)]['firehazard'] = 1;

    return G

def Generate_FireHazard_environment_complecated():
    # represent a level of LabRecruits game as a graph.
    G = nx.DiGraph()
    # representing a room as a node in the graph
    G.add_node(0)
    G.add_node(1)
    G.add_node(2) # final destination
    G.add_node(3)
    G.add_node(4)
    G.add_node(5)
    G.add_node(6)
    G.add_node(7)
    G.add_node(8)
    G.add_node(9)
    G.add_node(10)
    G.add_node(11)
    G.add_node(12)
    G.add_node(13)

    #representing the doors as edges (bidirectional means connection from both ways)
    G.add_edges_from([(0, 1), (0, 3), (0, 8)])
    G.add_edges_from([(1, 0), (1, 7)])
    G.add_edges_from([(2, 6),(2, 7)])
    G.add_edges_from([(3, 0), (3, 4), (3, 5)])
    G.add_edges_from([(4, 1), (4, 3), (4, 5), (4, 8), (4, 9)])
    G.add_edges_from([(5, 3), (5, 4), (5, 6), (5, 10)])
    G.add_edges_from([(6, 2), (6, 5), (6, 12), (6, 13)])
    G.add_edges_from([(7, 1), (7, 2)])
    G.add_edges_from([(8, 0), (8, 4)])
    G.add_edges_from([(9, 4)])
    G.add_edges_from([(10, 5), (10, 11)])
    G.add_edges_from([(11, 10), (11, 12), (11, 13)])
    G.add_edges_from([(12, 6), (12, 11)])
    G.add_edges_from([(13, 11), (13, 6)])

    for n in G.nodes:
        G.nodes[n]['door'] = {g: 0 for g in G.out_edges(n)}  # 0 means door is closed
        G.nodes[n]['button'] = {g: 0 for g in G.out_edges(n)}  # 0 means button is not pressed
        G.nodes[n]['name'] = 'room' + str(n)
        #G.nodes[n]['firehazard'] = 0  # initially no firehazard

    # adding danger level as the number of firehazard present in the path (edge) from one room to another (so we consider
    # firehazard to be presented in the path from one room to another that means an edge should contain this info)
    G.edges[(0, 1)]['firehazard'] =7;
    G.edges[(0, 3)]['firehazard'] = 0;
    G.edges[(0, 8)]['firehazard'] = 0;
    G.edges[(1, 0)]['firehazard'] = 7;
    G.edges[(1, 7)]['firehazard'] = 7;
    G.edges[(2, 6)]['firehazard'] = 1;
    G.edges[(2, 7)]['firehazard'] = 5;
    G.edges[(3, 0)]['firehazard'] = 0;
    G.edges[(3, 4)]['firehazard'] = 1;
    G.edges[(3, 5)]['firehazard'] = 1;
    G.edges[(4, 1)]['firehazard'] = 1;
    G.edges[(4, 3)]['firehazard'] = 1;
    G.edges[(4, 5)]['firehazard'] = 1;
    G.edges[(4, 8)]['firehazard'] = 1;
    G.edges[(4, 9)]['firehazard'] = 0;
    G.edges[(5, 3)]['firehazard'] = 1;
    G.edges[(5, 4)]['firehazard'] = 1;
    G.edges[(5, 6)]['firehazard'] = 1;
    G.edges[(5, 10)]['firehazard'] = 1;
    G.edges[(6, 5)]['firehazard'] = 1;
    G.edges[(6, 2)]['firehazard'] = 1;
    G.edges[(6, 12)]['firehazard'] = 1;
    G.edges[(6, 13)]['firehazard'] = 1;
    G.edges[(7, 1)]['firehazard'] = 7;
    G.edges[(7, 2)]['firehazard'] = 5;
    G.edges[(8, 0)]['firehazard'] = 1;
    G.edges[(8, 4)]['firehazard'] = 1;
    G.edges[(9, 4)]['firehazard'] = 1;
    G.edges[(10, 5)]['firehazard'] = 1;
    G.edges[(10, 11)]['firehazard'] = 1;
    G.edges[(11, 10)]['firehazard'] = 1;
    G.edges[(11, 12)]['firehazard'] = 1;
    G.edges[(11, 13)]['firehazard'] = 1;
    G.edges[(12, 11)]['firehazard'] = 0;
    G.edges[(12, 6)]['firehazard'] = 0;
    G.edges[(13, 11)]['firehazard'] = 0;
    G.edges[(13, 6)]['firehazard'] = 0;


    return G

# ----------------------------------------------------------------------------------------------------------
def Generate_LargeMaze():
    NumberOfRoom =35
    # represent a level of LabRecruits game as a graph.
    G = nx.DiGraph()
    # representing a room as a node in the graph
    i=0
    for i in range(NumberOfRoom):
        G.add_node(i)
    #representing the doors as edges (bidirectional means connection from both ways)
    G.add_edges_from([(0, 1), (0, 19), (0, 7)])
    G.add_edges_from([(1, 0), (1, 7), (1, 19), (1, 2)])
    G.add_edges_from([(2, 1),(2, 26)])
    G.add_edges_from([(3, 2), (3, 4), (3, 19)])
    G.add_edges_from([(4, 3), (4, 5), (4, 20)])
    G.add_edges_from([(5, 4), (5, 6), (5, 27)])
    G.add_edges_from([(6, 5), (6, 21), (6, 18)])
    G.add_edges_from([(7, 0), (7, 1), (7, 8)])
    G.add_edges_from([(8, 7), (8, 9), (8, 28)])
    G.add_edges_from([(9, 8),(9, 10),(9, 22), (9, 29)])
    G.add_edges_from([(10, 9), (10, 11), (10, 22)])
    G.add_edges_from([(11, 10), (11, 12), (11, 23)])
    G.add_edges_from([(12, 11), (12, 13), (12, 29)])
    G.add_edges_from([(13, 12), (13, 14), (13, 31)])
    G.add_edges_from([(14, 13), (14, 15), (14, 24)])
    G.add_edges_from([(15, 14), (15, 16), (15, 25)])
    G.add_edges_from([(16, 15), (16, 17), (16, 25), (16, 31)])
    G.add_edges_from([(17, 16), (17, 18), (17, 33)])
    G.add_edges_from([(18, 17), (18, 6), (18, 21)])
    G.add_edges_from([(19, 0), (19, 1), (19, 3)])
    G.add_edges_from([(20, 4), (20, 21)])
    G.add_edges_from([(21, 6), (21, 18), (21, 20)])
    G.add_edges_from([(22, 9), (22, 10), (22, 23)])
    G.add_edges_from([(23, 11), (23, 22), (23, 24)])
    G.add_edges_from([(24, 14), (24, 23), (24, 25)])
    G.add_edges_from([(25, 15), (25, 16), (25, 24),(25, 34)])
    G.add_edges_from([(26, 2), (26, 27), (26, 28)])
    G.add_edges_from([(27, 5), (27, 26), (27, 33)])
    G.add_edges_from([(28, 8), (28, 26), (28, 30)])
    G.add_edges_from([(29, 9), (29, 12), (29, 30)])
    G.add_edges_from([(30, 28), (30, 29), (30, 32)])
    G.add_edges_from([(31, 13), (31, 16), (31, 32)])
    G.add_edges_from([(32, 30), (32, 31), (32, 33)])
    G.add_edges_from([(33, 17), (33, 27), (33, 32)])
    G.add_edges_from([(34, 25)])
    G.add_edges_from([(34, 25)])




    for n in G.nodes:
        G.nodes[n]['door'] = {g: 0 for g in G.out_edges(n)}  # 0 means door is closed
        G.nodes[n]['button'] = {g: 0 for g in G.out_edges(n)}  # 0 means button is not pressed
        G.nodes[n]['name'] = 'room' + str(n)
        #G.nodes[n]['firehazard'] = 0  # initially no firehazard

    for e in G.edges:
        G.edges[e]['firehazard']=0
    return G


# ----------------------------------------------------------------------------------------------------------
def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G,pos)
    plt.show()


# ----------------------------------------------------------------------------------------------------------
#function -  get all shortest path
def getAllPairShortestPaths(G):
    nodes=list(G.nodes())
    print(nx.is_strongly_connected(G))
    # print(nx.strongly_connected_components)
    allShortestPaths={}
    print('starting shortest path calculation : ')
    for n1 in nodes:
        for n2 in nodes:
            if (nx.has_path(G,n1,n2)==True):
                allShortestPaths[(n1,n2)]=list(nx.all_shortest_paths(G,n1,n2,weight=None))
            else:
                print('No path available between ', n1, ' and ',n2 )
                #allShortestPaths[(n1, n2)] =0
    print('end shortest path calculation')
    return allShortestPaths
#-----------------------------------------------------------------------------------------------------------
#function -  get all shortest path
def getShortestPaths(n1, n2):
    allShortestPaths={}
    print('starting shortest path calculation : ')
    #if (nx.has_path(G, n1, n2) == True):
    allShortestPaths[(n1, n2)] = list(nx.all_shortest_paths(G, n1, n2, weight=None))
    #else:
    #    print('No path available between ', n1, ' and ', n2)
    print('end shortest path calculation')
    return allShortestPaths


# ----------------------------------------------------------------------------------------------------------
#check if the current state is the terminal state
def isterminalState(currentposition):
    #print('is terminal state', str(currentposition), str(goalState))
    if str(currentposition) == str(goalState):
        return True
    else:
        return False
# ----------------------------------------------------------------------------------------------------------
# get the qvalue for a (state,action) pair
def getQ(G, qtable, state_pos, state, action):
    qval=0
    #if this is a new state, i.e., there is no entry for this state in the Qtable, then add this state with 0 value for actions
    if state in qtable.keys():
        #print('state found =  ', state)
        for key in qtable[state].keys():
            if key==action:
                qval = qtable[state][key]
                #print('in getQ function -key, action , value', key, action, qtable[state][key])
    return qval

# ----------------------------------------------------------------------------------------------------------
# get the max qvalue for a (state,action) pair
def maxQ(qtable, state):
    maxqval=0
    if state in qtable.keys():
        for key in qtable[state].keys():
            all_qvalues = qtable[state].values()
            maxqval = max(all_qvalues)
    return maxqval

# ----------------------------------------------------------------------------------------------------------
# entry in qtable
def setqtableentry(G, state_pos,state,qtable, actionlist):
    if state not in qtable.keys(): #new state, add this in the qtable
        #print('adding this new state')
        qtable[state] = {}
        for i in range(len(actionlist)):
            qtable[state][actionlist[i]] = 0
    return qtable
# ----------------------------------------------------------------------------------------------------------
#get the best action from a state looking at the qtable
def getbestaction(qtable,state):
    bestaction=''
    if state in qtable.keys():
        #print('get best act, funciton , state = ', state)
        #print('max val(state, action)', qtable[state])
        max_key = max(qtable[state], key=qtable[state].get)
        bestaction = max_key
        #print('best action ', bestaction)
    else:
        print('state not found')
    return bestaction

# ----------------------------------------------------------------------------------------------------------
#get the best action from a state looking at the qtable
def getapproximatebestaction(G,current_pos,qtable,state):
    #print('func approx, current pos = ', current_pos)
    max = 0
    bestaction=''
    if state in qtable.keys():
        #print('state , qval =', qtable[state])
        for objkey in qtable[state].keys():
            #print('considering action,val=',objkey, qtable[state][objkey])
            x = objkey.split("-")
            # print('split', x[0], x[1],x[2])
            key1 = str(x[1])  # int(x[1])
            key2 = str(x[2])  # int(x[2])
            if (G.has_edge(key1, key2)):
                #print('has edge = ')
                if (qtable[state][objkey]> max):
                    max = qtable[state][objkey]
                    bestaction = objkey
        #print('state = ', state)
        #print('max action val = ', max, bestaction)
        #print('max val(state, action)', qtable[state])
        #max_key = max(qtable[state], key=qtable[state].get)
        #bestaction = max_key
    #print('best action ', bestaction)
    else:
        print('state not found')
    return bestaction

# ----------------------------------------------------------------------------------------------------------
# choose an action
def getAction(G,possibleActionList,epsilon,currentState,qtable,epsilonGreedyAlgo):
    #print("Inside function : getAction()")
    rval = np.random.uniform(0, 1)
    # select an action : act randomly sometimes to allow exploration
    if rval < epsilon:# Explore a random action
        #print('Random value and epsilon value = ',  rval,  epsilon,"  Decision: Explore, take an action random from possible list")
        action = random.choice((possibleActionList))  # action = env.randomAction()
    # if not select max action in Qtable (act greedy)
    else:  # Use the action with the highest q-value
        # print('greedy')
        if (epsilonGreedyAlgo == True): # follow the eplison greedy algorithm
            action = getbestaction(qtable, currentState)
            #print('Random value and epsilon value = ',  rval,  epsilon, " Decision: Exploit, take best action so far from Q-table")
        else:  # follow pure random exploration
            action = random.choice((possibleActionList))
            #print('Random explore - Random value and epsilon value = ', rval, epsilon, "  Decision: Explore, take an action random from possible list")

    return action

# ----------------------------------------------------------------------------------------------------------
def getState(G,node_pos,qtable,training):
    #print("Inside function getState()")
    state =""
    possibleActionList =[]
    # get the number of doors possible to go from this room
    #print('room position= ', node_pos)
    #print('get door status of this state = ', G.nodes[node_pos]['door'])
    #print('get button status of this state = ', G.nodes[node_pos]['button'])
    for key in G.nodes[node_pos]['door'].keys():
        # d means door, following current and connecting room position, and its status (open -1, closed-0).Example : d-2-3-0 (i.e., it is a door from room 2 to room 3. The status of the door is closed)
        doorname =  'd-'+ str(node_pos)+'-'+str(key[1])
        state = state +" "+doorname+'-'+str(G.nodes[node_pos]['door'][key])
        possibleActionList.append(doorname)

    for key in G.nodes[node_pos]['button'].keys():
        # b means door, following current and connecting room position, and its status (pressed -1, not pressed-0).Example : b-2-3-0 (i.e., it is a button for a door from room 2 to room 3. The status of the button is not pressed)
        buttonname = 'b-' + str(node_pos) + '-' + str(key[1])
        state = state + " " + buttonname + '-' + str(G.nodes[node_pos]['button'][key])
        possibleActionList.append(buttonname)

    #for key in G.nodes[str(node_pos)]['button'].keys():
    #    buttonname = 'b-' + str(node_pos)+'-'+str(key[1])      # b means door, following current and connecting room position, and its status (pressed -1, not pressed-0).Example : b-2-3-0 (i.e., it is a button for a door from room 2 to room 3. The status of the button is not pressed)
    #    state = state + " " + buttonname+'-'+str(G.nodes[str(node_pos)]['button'][key])
    #    possibleActionList.append(buttonname)

    #if (G.nodes[str(node_pos)]['firehazard']== 1):
        # firehazard is an element inside a room. so the presentation is fz means firehazard, follows by the room number and again the room number, then always 1 indicating it exists
    #    firehazard = 'fz-' + str(node_pos)+'-'+str(node_pos)
    #    state = state + " " + firehazard+'-1'
    #    possibleActionList.append(firehazard)

    state =state.lstrip()
    #print("in getState(), State  = ", state)
    if (training == True):  # training time
        qtable = setqtableentry(G, node_pos, state, qtable, possibleActionList)
    return state, possibleActionList
#-----------------------------------------------------------------------------------------------------------
def getStateDistance(currentState, nextState):  # calculate jaccard similarity
    #print("In func getstatedistance(), current state = ", currentState, "  next state = ", nextState)
    statesimilaritymeasure = 0
    x = currentState.split(" ")
    nextst = nextState.split(" ")
    commonelement=0
    totalelement=0
    for i in range(len(x)):
        if (x[i] in nextState):
            commonelement =commonelement+1
    totalelement = (len(nextst) + len(x)) -  commonelement
    if (commonelement>0):
        statesimilaritymeasure = commonelement/totalelement
    #print("dissimilarity val = ",(1-statesimilaritymeasure), "  common = ", commonelement, "  total = ", totalelement)
    return (1-statesimilaritymeasure)
#-----------------------------------------------------------------------------------------------------------
def calculateStateVisitFrequency(nextState, exploredstateos):
    freqv=1
    print(exploredstateos)
    if nextState in exploredstateos:  # calculate frequency of visit of this state
        freqv = freqv + exploredstateos[nextState]
        #print("In Fun calculateStateVisitFrequency() -  this state is visited before, freq considering current occ = ",freqv)
    return freqv
#-----------------------------------------------------------------------------------------------------------
def getReward(G,currentState, nextState, current_pos, next_state_pos, exploredstateos):
    #print("Insider func getReward(), From : "+currentState,"  To: ", nextState)
    #print("Explored state = ", exploredstateos)
    global CurrentHealthPoint
    global FlagAgentDied
    reward=0
    flaghealthloss=False
    edgeHealthlossthreshold =15 #3 (3*5) firehazard in that path/edge

    if (next_state_pos == goalState):  # get max reward for reaching the final destination
        reward = reward + maxreward
        print("Reach Final Destination - Get max reward = ", reward)
    else:
        # First consider the reward/penalty due to firehazard : case 1 : Reward for taking safer route (saving health point), penalty for taking route with high danzer level (loosing health point)
        if (current_pos != next_state_pos):  # if the agent move to a new place then calculate the danger of that path
            firehazardnum = G.edges[(current_pos, next_state_pos)]['firehazard']
            print("Number of firehz in this path ", firehazardnum)
            healthloss = firehazardnum * danzerVal
            # healthstatus = CurrentHealthPoint - healthloss
            currhealt = CurrentHealthPoint  # CurrentHealthPoint  # get current health value
            # print("Danger case : Previous health value = ", CurrentHealthPoint)
            currhealt = (currhealt - healthloss)
            setCurrentHealthStatus(currhealt)  # update health value
            # print("Health loss = ", healthloss, "Current health point = ", CurrentHealthPoint)
            if (CurrentHealthPoint < HealthPointThreshold):  # if health point below a threshold
                if(healthloss > edgeHealthlossthreshold):  # if agent loss health point particularly for traversing this edge/door
                    reward = reward - (FullHealthPoint - currhealt) #-- TODO -need to consider
                    flaghealthloss = True
                    print("Danger case : Penalty for going health point below threshold, penalty = ", (FullHealthPoint - currhealt), "  final penalty = ", reward)

        if (CurrentHealthPoint <= 0):  # if agent is dead
            # print('Before - Agent dead flag = ', FlagAgentDied)
            setAgentDeadFlag(True)
            flaghealthloss= True
            reward = reward - (100 * 100)  # agent is dead, big penalty
            print("Danger case : Huge penalty as agent died = ", reward, "  After - agent dead flag = ", FlagAgentDied)
        #print("This state is not destination. Now calculate reward or penalty for this state")

        # only count this case if no penalty is given for health loss. Meaning if significant health loss is occured then a path is not encouraged even if it is novel
        # case 2 : Reward and penalty for curiosity - Encourage exploring - curiosity driven (reward for visiting new state, penalty for visiting a state n times)
        if (flaghealthloss== False):
            dissimilarity = getStateDistance(currentState, nextState)  # get similarity between two states
            statevisitfrequency = calculateStateVisitFrequency(nextState, exploredstateos)
            exploredstateos[nextState] =statevisitfrequency  # update state visit frequency

            #a) Reward if it is a new state compare to the previous state and this state is not visited much
            if (dissimilarity >=0.2 and statevisitfrequency<=statevisitthreshold):
                reward = reward + (dissimilarity * 10)
                #print("curiosity case: Reward for visiting a new state, distance from prev state = ", dissimilarity, "  visit freq = ", statevisitfrequency)

            # b) Penalty for exploring the same state more than threshold time
            if (statevisitfrequency > statevisitthreshold):
                reward = reward -(dissimilarity*10 + StateVisitPENALTY)
                #print("curiosity case: Penalty for visiting same state, distance from prev state = ", dissimilarity, "  visit freq = ", statevisitfrequency)



        #print("Info about agent, health point =", CurrentHealthPoint, "  state dissimilarity = ", dissimilarity, "state visit freq = ", statevisitfrequency)
    print("Final reward after executing this action = ", reward)
    return reward
# -----------------implementation of sparse reward scenario (used by RL sparse)-----------------------------------------------------------------------------------------
# get reward only for reaching the final state and get penalty for health loss
def getRewardSparse(G,currentState, nextState, current_pos, next_state_pos,flagRandom):
    print("Insider func getRewardSparse(), From : "+currentState,"  To: ", nextState)
    global CurrentHealthPoint
    global FlagAgentDied
    reward=0
    flaghealthloss=False
    edgeHealthlossthreshold =15 #3 (3*5) firehazard in that path/edge

    if (next_state_pos == goalState):  # get max reward for reaching the final destination
        reward = reward + maxreward
        print("SparseReward - Reach Final Destination - Get max reward = ", reward)
    else:
        if (current_pos != next_state_pos):  # if the agent move to a new place then calculate the danger of that path
            firehazardnum = G.edges[(current_pos, next_state_pos)]['firehazard']
            print("Number of firehz in this path ", firehazardnum)
            healthloss = firehazardnum * danzerVal
            # healthstatus = CurrentHealthPoint - healthloss
            currhealt = CurrentHealthPoint  # CurrentHealthPoint  # get current health value
            # print("Danger case : Previous health value = ", CurrentHealthPoint)
            currhealt = (currhealt - healthloss)
            setCurrentHealthStatus(currhealt)  # update health value
            # print("Health loss = ", healthloss, "Current health point = ", CurrentHealthPoint)
            if (CurrentHealthPoint < HealthPointThreshold):  # if health point below a threshold
                if (healthloss > edgeHealthlossthreshold):  # if agent loss health point particularly for traversing this edge/door
                    reward = reward - (FullHealthPoint - currhealt)  # -- TODO -need to consider
                    flaghealthloss = True
                    print("Sparse reward - Danger case : Penalty for going health point below threshold, penalty = ",
                          (FullHealthPoint - currhealt), "  final penalty = ", reward)

        if (CurrentHealthPoint <= 0):  # if agent is dead
            # print('Before - Agent dead flag = ', FlagAgentDied)
            setAgentDeadFlag(True)
            flaghealthloss = True
            reward = reward - (100 * 100)  # agent is dead, big penalty
            print("Sparse reward - Danger case : Huge penalty as agent died = ", reward, "  After - agent dead flag = ",FlagAgentDied)
        if(flagRandom==True):
            reward=0

        print("Sparse/Random reward - Final reward after executing this action = ", reward)
    return reward

# ----------------------------------------------------------------------------------------------------------
def executeAction(G,pos,action):
    #print("Inside function executeAction()")
    nextstatepos =-1
    #rewardval =-1
    x = action.split("-")
    #print("Action = ", action," After split", x[0], x[1],x[2])
    key1 =int(x[1])#str(x[1])#
    key2 =int(x[2]) #str(x[2])#
    if (x[0] =='d'):
        if (G.nodes[pos]['door'][(key1,key2)]==1):  # if the door is open then we can go to the connecting door
            nextstatepos = key2  # next room position
            #print("Action : DOOR View status. Door is open = ", G.nodes[pos]['door'][(key1, key2)], "going to next room= ", nextstatepos)
        if (G.nodes[pos]['door'][(key1,key2)]==0):  # door is closed, nothing to be done. First need to press the button that opens this door
            nextstatepos = key1  # staying in the current room
            #print("Action : DOOR View status. Door is closed = ", G.nodes[pos]['door'][(key1, key2)], "Staying same room= ", nextstatepos)
    if (x[0] == 'b'):
        #print('Action : Interact with Button. before status,(k1,k2) =',key1, key2, "  button = ",   G.nodes[pos]['button'][(key1, key2)], "  door = ", G.nodes[pos]['door'][(key1, key2)])
        #interacting a button means change the status of this button and doors associated with it
        nextstatepos = key1  # staying in the current room, interacting with a button does not mean moving to a new place
        #interaction with a button means changingm status of a button and doors associated with it
        if (G.nodes[pos]['button'][(key1,key2)]==1):  # if the switch is already pressed then pressing again will change the status of this switch  again
            G.nodes[pos]['button'][(key1, key2)] = 0 #change status of the button
            # change status of door
            if (G.nodes[pos]['door'][(key1, key2)] == 0):
                G.nodes[pos]['door'][(key1, key2)] = 1  # door is open
            else:
                G.nodes[pos]['door'][(key1, key2)] = 0  # doogetr is open
            #print('After interaction status, button = ', G.nodes[pos]['button'][(key1, key2)], "  door = ",     G.nodes[pos]['door'][(key1, key2)], "  staying same room = ", nextstatepos)
        else:  # if the button is not pressed, press the button to open the connecting door
            G.nodes[pos]['button'][(key1, key2)] = 1  # change the button state as pressed, do does the door state as open
            if (G.nodes[pos]['door'][(key1, key2)] == 0):
                G.nodes[pos]['door'][(key1, key2)] = 1  # door is open
            else :
                G.nodes[pos]['door'][(key1, key2)] = 0  # door is open
            #print('After interaction status, (k1,k2) =  ',key1, key2, "  button =  ",  G.nodes[pos]['button'][(key1, key2)], "  door = ", G.nodes[pos]['door'][(key1, key2)], "  staying same room = ", nextstatepos)

    return nextstatepos #, rewardval

# ----------------------------------------------------------------------------------------------------------
#Training - Q-learning Algorithm
def QlearningTraining(G,numTrainingEpisodes, epsilonGreedyAlgo):
    # Initial training parameters
    alpha = 0.3   #learning_rate
    gamma = 0.7   #discount_factor
    epsilon = 0.8
    decay = 0.995
    print("Starting Q learning, epsilon = ", epsilon, " Decay step = ", decay)
    # for episode statistics
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    qtable ={}  #initialize qtable
    # start the learning episodes
    for episode in range(numTrainingEpisodes):
        print('----------------------------------Starting episode = ', (episode+1),"---------------------------")
        exploredstate={}
        ActionSeq=[]
        RewardSeq=[]
        print("Initializing explored state memory for this episode, now size = ", len(exploredstate))
        setAgentDeadFlag(False)      # indicate agent is alive
        setCurrentHealthStatus(100)  # set health point to full (100)
        print('Agent life status = ', FlagAgentDied)

        episode_reward = 0  # reward sequence for this episode
        # reset environment for this episode /create a copy of the environment
        G_={}
        G_ = copy.deepcopy(G)

        # Pick up a state randomly as a starting state for the agent in a learning episode
        current_pos = 0;# fixing initial position of the agent at room 0 #np.random.randint(1, (G_.number_of_nodes()+1))#np.random.randint(0, G_.number_of_nodes())  # Python excludes the upper bound
        currentState, possibleActionList = getState(G_,current_pos, qtable, True) # store possible list of actions from current state
        startpos = current_pos

        print("Initial position of agent at room  = ",current_pos, "  Current view of agent/state = ",currentState, "  Possible action list  = ", possibleActionList)
        print('Q-table at the begining of episode: ')
        printQtable(qtable)
        print('Starting position =', current_pos, ',  state =',currentState, '   possible action list  = ', possibleActionList)

        exploredstate[currentState] = 1  # store first state
        step = 1  # step counter require to finish an episode
        #pathlength =0

        while (isterminalState(current_pos) == False):  # loop for actions to achieve goal for this episode
            print("Num of step = ", step)
            step = step + 1
            if (flagRandom == True):
                print("Random Action")
                action = random.choice((possibleActionList))
            else:
                print("RL Action")
                action = getAction(G_, possibleActionList, epsilon, currentState, qtable, epsilonGreedyAlgo)  # get action

            next_state_pos= executeAction(G_,current_pos, action)  # execute action and get next step
            nextState, possibleActionList = getState(G_, next_state_pos, qtable, True)  # get next step
            if (flagSparseOrRandom == False):
                rewardval =getReward(G_,currentState, nextState, current_pos, next_state_pos, exploredstate)
            else:
                rewardval = getRewardSparse(G_, currentState, nextState, current_pos, next_state_pos, flagRandom)
            print("Action: ", action, "From : " + currentState, " To: ", nextState, "  Reward= ", rewardval,"  CurPos= ", current_pos, "  NextPos= ", next_state_pos)

            # store actions and rewards of this episode
            ActionSeq.append(action)
            RewardSeq.append(rewardval)

            episode_reward += rewardval  # cumulating reward for an episode
            #print("Action: ", action, "Current room pos = ", current_pos , "  Next position = ",next_state_pos)

            # get the current q value
            qval = getQ(G_, qtable, current_pos, currentState, action)
            max_value = maxQ(qtable, nextState)
            new_q_value = (1 - alpha) * qval + alpha * (rewardval + gamma * max_value)
            print('qval = ', qval, '  maxval = ', max_value, ' new qval = ', new_q_value)

            qtable[currentState][action] = new_q_value  # update q table
            currentState = nextState  # assign next state as current state of the agent
            current_pos = next_state_pos  # assign next room position as current room position of the agent

            if(isterminalState(current_pos)==True):   # Episode ending condition 1: if agent reaches its destination
                print('Agent reach destination, Agent status = ', FlagAgentDied)
                #print('Shortes path = ', list(nx.all_shortest_paths(G_, str(startpos), str(goalState), weight=None)), '   explored room = ', exploredrooms)

            if (step > maxStepPerEpisode): # Episode ending condition 2: if agent performs max steps allowed for an episode
                print('Terminating ..more than max steps per episode =',step, "  maxStepPerEpisode = ", maxStepPerEpisode)
                #print('shortes path = ', list(nx.all_shortest_paths(G_, str(startpos), str(goalState), weight=None)), '   explored room = ', exploredrooms)
                break

            if(FlagAgentDied == True): # Episode ending condition 1: if agent died
                print("Agent is dead, Ending episode", FlagAgentDied)
                break

        print("Number of steps taken = ", step)
        print("Action seq= ",ActionSeq)
        print("Reward seq= ", RewardSeq)
        #if (FlagAgentDied == False):
        # print("Agent alive after episode")
        #Decayed epsilon - Update epsilon value
        print("Reducing epsilon value, Before value  = ", epsilon)
        epsilon = epsilon * decay   # reduce epsilon value to reduce the exploration while encouraging exploitation in the next episode
        print("After allowing decay of epsilon, value = ", epsilon)
        ep_rewards.append(episode_reward)

    return qtable
# ----------------------------------------------------------------------------------------------------------
#print final Qtable
def printQtable(qtable):
    print('printing Q table')
    for state in qtable.keys():
        print('State : ', state)
        for key in qtable[state].keys():
            print('(statepos,action) = ',key, '   val = ',qtable[state][key])

# ----------------------------------------------------------------------------------------------------------
#test the agent
def TestAgent(G,qtable):
    print("----------------Testing agent ----------------------------------")
    penalties, edgecount = 0, 0
    ActionSeq = []
    RewardSeq = []
    exploredstate = {}
    path, pathedge="", ""

    current_pos = initialState
    currentState, possibleactionlist = getState(G,current_pos, qtable, False)
    path = path + str(current_pos)+' '
    pathedge = pathedge + str(current_pos) + ' '
    exploredstate[currentState] = 1  # store first state

    # set health point to full
    setAgentDeadFlag(False)  # indicate agent is alive
    setCurrentHealthStatus(100)  # set health point to full (100)

    while (isterminalState(current_pos)==False):
        action = getbestaction(qtable,currentState)
        #action = getapproximatebestaction(G, current_pos, qtable, currentState)
        next_state_pos = executeAction(G, current_pos, action)  # execute action and get next step
        nextState, possibleActionList = getState(G, next_state_pos, qtable, False)  # get next step
        rewardval = getReward(G, currentState, nextState, current_pos, next_state_pos, exploredstate)
        print('CurrPos= ', current_pos, ',  state= ', currentState, " Action : ",action, "next_pos =", next_state_pos, "  nextSt= ", nextState)
        path = path+' '+str(current_pos) + ' '+ str(action)+' ' + str(next_state_pos)

        if (current_pos != next_state_pos):
            pathedge = pathedge + str(next_state_pos) + ' '
            edgecount =edgecount+1

        currentState  = nextState
        current_pos = next_state_pos
        ActionSeq.append(action)
        RewardSeq.append(rewardval)

        if rewardval < 0:
            penalties += 1

        print("visited edges/room = ", pathedge)
        print("Action seq = ", ActionSeq)
        print("Reward seq = ", RewardSeq)
        print("Total Reward = ", sum(RewardSeq), "   Number of time agent gets penalty = ", penalties )

        print('Health point = ', CurrentHealthPoint)
        print('Agent life status = ', FlagAgentDied)
        #print('shortest path =', list(nx.all_shortest_paths(G, str(startpos), str(goalState), weight=None)) )


def plotlargegraph(G):
    pos = nx.spring_layout(G)  # G is my graph
    nx.draw(G, pos, node_color='#A0CBE2', edge_color='#BB0000', width=2, edge_cmap=plt.cm.Blues, with_labels=True)
    # plt.show()
    plt.savefig("graph.png", dpi=500, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)

def plot_degree_dist(G):
    #degrees = [G.degree(n) for n in G.nodes()]
    degrees = [G.degree(n) for n in G.nodes()]
    print('min degree = ',  min(degrees))
    plt.hist(degrees)
    plt.show()
# ----------------------------------------------------------------------------------------------------------
def checkshortestpathlength(G, initialstate):
    print('CHECK SHORTES PATH LENGTH')
    shortestpahtlength=[]
    #allShortestPaths={}
    for i in range(len(initialstate)):
        #print('starting  = ', initialstate[i])
        #if (nx.has_path(G,str(startingnode),str(goalState))==True):
        shortestpahtlength.append(nx.shortest_path_length(G, source=str(initialstate[i]), target=str(goalState)))
            #print('start = ', startingnode, '  end = ', goalState, ' path length =  ',nx.shortest_path_length(G, source=startingnode, target=goalState))
            #allShortestPaths[(startingnode,goalState)]=list(nx.all_shortest_paths(G,startingnode,goalState,weight=None))
        #else:
        #    print('No path available between ', startingnode, ' and ',goalState )
                #allShortestPaths[(n1, n2)] =0
    avgpathlength =-1
    avgpathlength = np.average(shortestpahtlength)
    #print('path from 757 room= ',list(nx.all_shortest_paths(G,757,goalState,weight=None)))
    print('shortest path length = ', shortestpahtlength)
    print('Average path length to reach destination = ', avgpathlength)
    return avgpathlength

# ----------------------------------------------------------------------------------------------------------
def loadButtonInformation(filepath):
    buttonmappint = {}
    print('load button -  reading csv file, opening')
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        linenum = 0
        for row in reader:
            linenum = linenum + 1
            # print('line num = ', linenum)
            # parsing each column of a row
            colnum = 0
            key = ''
            for col in row:
                colnum = colnum + 1
                if (colnum == 1):
                    strkey = str(col)
                    if (strkey.startswith('b')==False):
                        break
                    key = col
                    buttonmappint[key] = {'doorlist':[], 'status': 0}
                    # print('key =  ', col, key, '  ')
                else:
                    buttonmappint[key]['doorlist'].append(col)
    #print('buttonmap ', buttonmappint)
    return buttonmappint

# ----------------------------------------------------------------------------------------------------------
# Main function -- Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generategraph = True  # true : generate graph, false : load graph from file
    training = True   # true : run training session, false : load trained q-table
    createtestset = False  # true: create the test set indicating starting point for the agent, false :

    # define path of the output directory where the environment graph and trained Q-table is located
    qtableoutputpath = 'results/Qtable'
    graphpath = 'results/RL_test_57_rooms_63_doors.xml'
    csvfilepath ='results/RL_test_57_rooms_63_doors.csv'
    shortestpath = 'results/allshortestpaths'



    sys.stdout = open('outputrl.txt', 'w')
    # -------------------------------------------------------------------------------------------------
    # Get the environment, environment is represented as a graph, either generate it or load from file
    if (generategraph == True):
        #environment is a maze of n rooms. It is represented as a graph. Generate the graph
        G = {}
        G = Generate_LargeMaze()#Generate_FireHazard_environment() #Generate_FireHazard_environment()
        print('Number of nodes = ', G.number_of_nodes())
        print(G.nodes(data=True))
        #plot_graph(G)
        #exit()

    if (generategraph == False): # graph already exists in a .xml file, need to load it
        buttonMapping = loadButtonInformation(csvfilepath)
        G = nx.DiGraph()
        G = loadGraph(graphpath)
        print('number of nodes = ', G.number_of_nodes(), '  number of edges = ', G.number_of_edges())
        print('Edges =  ', G.edges())
        #plot_degree_dist(G)
        #plotlargegraph(G)
        print(G.nodes(data =True))
        #print('nodes ',G.number_of_nodes())
        #print(nx.degree_histogram(G))
        #print(nx.degree(G))
        #plot_graph(G)
        print('is connected = ', nx.is_connected(G))
        print('sorted degree = ', sorted(G.degree, key=lambda x: x[1], reverse=True))

    #exit()
    # --------------------------------------------------------------------------------------------------
    # Training phase : Q learning agent, a
    Q_table={}
    if (training == True): # run the training session
        print('Start training, to reach goal room = ', goalState)
        start = time.time()
        Q_table = QlearningTraining(G,numTrainingEpisodes, epsilonGreedyAlgo)
        print('Training time: ', (time.time() - start))
        print('-----------------------------------------------------------------')
        printQtable(Q_table)
        print("Q table size = ", len(Q_table.keys()))
        f = open(qtableoutputpath+'.pickle', 'wb') #store Q-table (after training) in a pickle file
        pickle.dump(Q_table, f)
        f.close()
        #exit()


    #print('number of edges = ', G.number_of_edges())
    if (training == False): # training has already done, only need to load the trained Q-table
        filename = 'results/Qtable50k.pickle'
        Q_table = pickle.load(open(filename,'rb'))
        print('Loading Q-table from file, file name : ', filename)
        #printQtable(Q_table)
        print('Number of state in Q-table = ',len(Q_table.keys()))
        #print('val = ', Q_table['d-14-236-1 d-14-474-0 d-14-786-1 d-14-807-0 d-14-825-0 b-14-236-1 b-14-474-0 b-14-786-1 b-14-807-0 b-14-825-0'])

    #allShortestPaths = list(nx.all_shortest_paths(G, '5', '18', weight=None))
    #print('Path from node 5 to node 18', allShortestPaths)
    #exit()

    #Testing

    TestAgent(G,Q_table)
    #TestCoverage()

    #for testing robustness - slightly changed graph from the one the agent is trained
    #TestAgent(G1, Q_table, numberoftestepisode, initialstate)
    print('Q table size = ', len(Q_table.keys()))
    sys.stdout.close()