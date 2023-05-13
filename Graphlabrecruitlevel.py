import numpy as np
import networkx as nx
import random
import copy
import time
import pickle
import matplotlib.pyplot as plt
import csv

goalState = 100  # identify the destination room
maxreward =100  # max reward for reaching the destination room
numOfFirehazard =20

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
        G.nodes[n]['button'] = {g: 0 for g in G.edges(n)}#{g: 0 for g in G.out_edges(n)}   # 0 means button is not pressed
        G.nodes[n]['name'] = 'room'+str(n)
        G.nodes[n]['firehazard'] = 0  # initially no firehazard

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
def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G,pos)
    plt.show()

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
    # select an action : act randomly sometimes to allow exploration
    if np.random.uniform(0, 1) < epsilon:# Explore a random action
        # print('exploit')
        action = random.choice((possibleActionList))  # action = env.randomAction()
    # if not select max action in Qtable (act greedy)
    else:  # Use the action with the highest q-value
        # print('greedy')
        if (epsilonGreedyAlgo == True): # follow the eplison greedy algorithm
            action = getbestaction(qtable, currentState)
        else:  # follow pure random exploration
            action = random.choice((possibleActionList))

    #print('action ', action)
    return action

# ----------------------------------------------------------------------------------------------------------
def getState(G,node_pos,qtable,training):
    state =""
    possibleActionList =[]
    # get the number of doors possible to go from this room
    #print('node pos= ', node_pos)
    #print('get state = ', G.nodes[str(node_pos)])
    for key in G.nodes[str(node_pos)]['door'].keys():
        # d means door, following current and connecting room position, and its status (open -1, closed-0).Example : d-2-3-0 (i.e., it is a door from room 2 to room 3. The status of the door is closed)
        doorname =  'd-'+ str(node_pos)+'-'+str(key[1])
        state = state +" "+doorname+'-'+str(G.nodes[str(node_pos)]['door'][key])
        possibleActionList.append(doorname)

    for key in G.nodes[str(node_pos)]['button'].keys():
        # b means door, following current and connecting room position, and its status (pressed -1, not pressed-0).Example : b-2-3-0 (i.e., it is a button for a door from room 2 to room 3. The status of the button is not pressed)
        buttonname = 'b-' + str(node_pos)+'-'+str(key[1])
        state = state + " " + buttonname+'-'+str(G.nodes[str(node_pos)]['button'][key])
        possibleActionList.append(buttonname)

    if (G.nodes[str(node_pos)]['firehazard']== 1):
        # firehazard is an element inside a room. so the presentation is fz means firehazard, follows by the room number and again the room number, then always 1 indicating it exists
        firehazard = 'fz-' + str(node_pos)+'-'+str(node_pos)
        state = state + " " + firehazard+'-1'
        possibleActionList.append(firehazard)

    state =state.lstrip()
    if (training == True):  # training time
        qtable = setqtableentry(G, node_pos, state, qtable, possibleActionList)
    return state, possibleActionList

# ----------------------------------------------------------------------------------------------------------
def executeAction(G,pos,action):
    nextstatepos =-1
    rewardval =-1
    x = action.split("-")
    #print('split', x[0], x[1],x[2])
    key1 =str(x[1])#int(x[1])
    key2 =str(x[2])#int(x[2])
    if (x[0] =='d'):
        #print('it is door, value ', G.nodes[pos]['door'][(key1, key2)])
        if (G.nodes[str(pos)]['door'][(key1,key2)]==1):  # if the door is open then we can go to the connecting door
            nextstatepos = key2  # next room position
            if (key2 == str(goalState)):
                rewardval = maxreward # get the reward from going to the connecting room
            else:
                rewardval = 0  # TO: to check if it learns the shortest path
            #print('Door is open, going to next room ', G.nodes[pos]['door'][(key1, key2)])
        if (G.nodes[str(pos)]['door'][(key1,key2)]==0):  # door is closed, nothing to be done. First need to press the button that opens this door
            nextstatepos = key1  # staying in the current room
            rewardval = 0  #NEED TO CHECK
            #print('Door is closed, first need to press the button that opens it ', G.nodes[pos]['door'][(key1, key2)])

    if (x[0] == 'b'):
        #print('it is button, value ', G.nodes[pos]['button'][(key1, key2)])
        if (G.nodes[str(pos)]['button'][(key1,key2)]==1):  # if the button is already pressed, do noting
            #print('button value = ', G.nodes[pos]['button'][(key1,key2)], 'already pressed')
            G.nodes[str(pos)]['door'][(key1, key2)] = 1  # refresh the status of the door, it should be open
            nextstatepos = key1  # next room position
            rewardval = 0
        if (G.nodes[str(str(pos))]['button'][(key1,key2)]==0):  # if the button is not pressed, press the button to open the connecting door
            #print('button value = ', G.nodes[pos]['button'][(key1, key2)], 'Not pressed')
            nextstatepos = key1  # staying in the current room
            rewardval = 0
            # change the button state as pressed, do does the door state as open
            G.nodes[str(pos)]['button'][(key1, key2)] = 1  # button is pressed
            G.nodes[str(pos)]['door'][(key1, key2)] = 1  # door is open
            #print('Changing button value ', G.nodes[pos]['button'][(key1, key2)])

    if (x[0] == 'fz'):
        nextstatepos = key1
        rewardval = -1000  #PENALTY IF INTERACT WITH FIREHAZARD
        #print('it is firehazard, value ', G.nodes[pos]['firehazard'])

    return nextstatepos, rewardval

# ----------------------------------------------------------------------------------------------------------
#Training - Q-learning Algorithm
def QlearningTraining(G,numTrainingEpisodes, epsilonGreedyAlgo):
    # Initial training parameters
    alpha = 0.1   #learning_rate
    gamma = 0.8   #discount_factor
    epsilon = 0.1
    decay = 0.1
    # for episode statistics
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    qtable ={}  #initialize qtable

    # start the learning episodes
    for episode in range(numTrainingEpisodes):
        exploredrooms=[]
        print('Starting episode = ', (episode+1))
        episode_reward = 0
        # reset environment for this episode /create a copy of the environment
        G_={}
        G_ = copy.deepcopy(G)

        # Pick up a state randomly as a starting state for the agent in a learning episode
        possibleActionList = []
        current_pos = np.random.randint(1, (G_.number_of_nodes()+1))#np.random.randint(0, G_.number_of_nodes())  # Python excludes the upper bound
        currentState, possibleActionList = getState(G_, current_pos, qtable, True)
        startpos = current_pos
        #exploredrooms.append(current_pos)
        #print('printing q table for the first time')
        # printQtable(qtable)
        #print('Starting position =', current_pos, ',  state =',currentState, '   possible action list  = ', possibleActionList)
        step = 1  # introducing loop breaking number
        samestate = 0
        #pathlength =0
        while (isterminalState(current_pos) == False):
            # print('\n---not final state, search again')
            step = step + 1
            action = getAction(G_, possibleActionList, epsilon, currentState, qtable, epsilonGreedyAlgo)
            next_state_pos, rewardval = executeAction(G_,current_pos, action)
            episode_reward += rewardval
            nextState, possibleActionList = getState(G_, next_state_pos, qtable, True)


            # check if the agent is roaming only between explored states
            if ((step > 3) and (isterminalState(current_pos) == False) and (next_state_pos in exploredrooms)):
            #if ((step >3) and (isterminalState(current_pos) == False) and (next_state_pos in exploredrooms)):
                rewardval =  rewardval - 1

            if (next_state_pos not in exploredrooms):
                exploredrooms.append(next_state_pos)


            # check if the agent staying same state for long, then it will get some penalty
            if ((isterminalState(current_pos) == False) and (next_state_pos == current_pos)):
            #if (nextState == currentState):
                #print('current state pos = ', current_pos, '   next state pos = ', next_state_pos)
                samestate = samestate + 1
            else:
                samestate = 0

            if (samestate > 4):  # penalty
                #print('-----PENALTY--------, in same state for consequtive time = ',samestate)
                rewardval = rewardval-1

            #print('next_state_pos =', next_state_pos, '   action = ', action, '  reward val = ', rewardval)
            #print('next state = ', nextState, '  possible action list= ', possibleActionList )

            # get the current q value
            qval = getQ(G_, qtable, current_pos, currentState, action)
            max_value = maxQ(qtable, nextState)
            new_q_value = (1 - alpha) * qval + alpha * (rewardval + gamma * max_value)
            #print('qval = ', qval, '  maxval = ', max_value, ' new qval = ', new_q_value)
            # update q table
            qtable[currentState][action] = new_q_value
            currentState = nextState
            current_pos = next_state_pos
            #print('new current pos = ', current_pos, 'current state = ', currentState)
            if(isterminalState(current_pos)==True):
                print('reach destination')
                print('shortes path = ', list(nx.all_shortest_paths(G_, str(startpos), str(goalState), weight=None)), '   explored room = ', exploredrooms)

            if (step > 5000):
                print('terminating ..more than 3000 steps.Start pos =',startpos)
                print('shortes path = ', list(nx.all_shortest_paths(G_, str(startpos), str(goalState), weight=None)), '   explored room = ', exploredrooms)
                break

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
def TestAgent(G,qtable,numberofepisodes, initialstate):
    # to check with identical initial state
    #initialstate=[0, 2, 6, 6, 4, 6, 5, 0, 0, 8, 7, 6, 9, 6, 6, 4, 8, 0, 2, 6, 7, 6, 5, 6, 1, 3, 5, 1, 6, 4, 0, 6, 7, 7, 7, 7, 1, 8, 0, 7, 3, 0, 1, 7, 0, 2, 3, 8, 2, 0, 6, 3, 6, 0, 2, 8, 1, 1, 1, 5, 7, 4, 6, 7, 7, 0, 6, 3, 6, 0, 8, 6, 5, 2, 2, 1, 0, 8, 6, 3, 0, 4, 7, 8, 7, 4, 7, 4, 3, 6, 2, 1, 8, 9, 8, 1, 2, 7, 5, 8, 8, 3, 7, 8, 9, 6, 9, 5, 6, 8, 2, 8, 2, 6, 1, 3, 2, 2, 5, 0, 6, 3, 0, 2, 2, 2, 3, 2, 8, 1, 7, 7, 7, 4, 4, 6, 4, 3, 5, 5, 9, 9, 7, 2, 2, 8, 6, 1, 1, 1, 6, 5, 7, 3, 7, 7, 7, 6, 3, 6, 4, 5, 3, 4, 9, 6, 0, 3, 0, 2, 5, 3, 5, 9, 5, 1, 6, 1, 6, 0, 4, 8, 2, 5, 8, 5, 1, 6, 2, 1, 3, 7, 7, 5, 0, 6, 1, 9, 3, 1, 7, 5, 9, 1, 7, 9, 5, 3, 7, 3, 2, 8, 5, 8, 7, 8, 4, 3, 1, 1, 7, 4, 5, 0, 0, 9, 6, 5, 3, 8, 8, 1, 5, 7, 6, 4, 8, 7, 5, 9, 2, 3, 2, 2, 0, 9, 0, 3, 5, 0, 8, 6, 4, 3, 3, 7, 4, 2, 4, 5, 4, 3, 9, 7, 2, 6, 9, 9, 2, 9, 6, 9, 2, 5, 9, 0, 3, 6, 4, 3, 2, 9, 3, 8, 9, 5, 0, 4, 3, 8, 9, 6, 7, 5, 1, 1, 2, 2, 0, 4, 9, 4, 2, 5, 4, 4, 3, 3, 2, 7, 6, 2, 5, 8, 8, 8, 5, 0, 5, 4, 1, 3, 7, 4, 7, 4, 1, 1, 2, 3, 7, 5, 3, 3, 9, 4, 5, 8, 0, 0, 2, 2, 9, 1, 6, 2, 0, 4, 1, 6, 4, 7, 2, 6, 4, 9, 5, 1, 2, 5, 6, 1, 7, 2, 5, 2, 3, 4, 3, 3, 5, 8, 9, 3, 2, 7, 0, 2, 6, 6, 3, 2, 3, 1, 9, 1, 8, 0, 0, 2, 9, 3, 0, 8, 4, 7, 5, 0, 3, 0, 9, 1, 1, 1, 5, 8, 0, 9, 3, 2, 8, 3, 4, 0, 2, 5, 5, 3, 2, 4, 8, 3, 0, 3, 5, 0, 2, 7, 0, 4, 4, 4, 4, 4, 2, 2, 9, 3, 8, 0, 1, 4, 0, 8, 6, 9, 2, 8, 6, 8, 2, 3, 0, 2, 0, 5, 0, 4, 3, 1, 5, 4, 9, 9, 1, 1, 3, 6, 8, 1, 1, 2, 5, 6, 5, 4, 4, 4, 1, 5, 8, 9, 4, 6, 7, 2, 4, 9, 0, 7, 9, 6, 3, 0, 1, 0, 2, 0, 5, 0, 3, 5, 6, 3, 6, 2, 9, 7, 3, 4, 4, 2, 4, 7, 7, 2, 6, 7, 6, 9, 9, 2, 0, 9, 2, 7, 1, 1, 3, 4, 8, 1, 9, 2, 0, 9, 9, 2, 7, 0, 6, 1, 6, 7, 2, 5, 3, 0, 5, 8, 5, 0, 3, 8, 9, 6, 1, 0, 4, 1, 5, 7, 1, 0, 7, 6, 2, 8, 6, 6, 8, 2, 1, 4, 4, 1, 5, 3, 9, 4, 9, 7, 9, 6, 0, 0, 1, 3, 3, 4, 0, 1, 6, 7, 4, 0, 2, 5, 2, 6, 9, 5, 1, 4, 9, 8, 2, 3, 9, 3, 2, 3, 8, 1, 8, 1, 1, 6, 1, 2, 5, 3, 5, 9, 2, 3, 5, 5, 8, 6, 3, 8, 8, 8, 8, 8, 7, 4, 8, 2, 1, 9, 9, 9, 9, 3, 5, 1, 1, 8, 0, 4, 7, 4, 1, 2, 3, 1, 3, 6, 1, 1, 0, 2, 6, 3, 9, 4, 5, 2, 8, 2, 6, 8, 9, 4, 0, 7, 9, 0, 3, 9, 0, 1, 7, 0, 9, 1, 1, 5, 2, 7, 7, 7, 3, 4, 2, 5, 3, 6, 9, 5, 1, 5, 7, 0, 2, 3, 7, 4, 2, 5, 4, 7, 0, 5, 3, 8, 0, 9, 9, 9, 9, 3, 9, 6, 2, 8, 4, 6, 8, 9, 1, 2, 1, 0, 0, 4, 3, 4, 9, 6, 1, 8, 4, 1, 0, 3, 1, 0, 4, 7, 7, 9, 5, 9, 3, 0, 2, 2, 3, 9, 7, 5, 9, 3, 8, 3, 9, 4, 3, 9, 5, 0, 9, 5, 0, 5, 0, 1, 0, 1, 3, 5, 5, 4, 2, 2, 9, 4, 6, 0, 5, 1, 7, 6, 9, 2, 4, 1, 4, 7, 0, 1, 2, 9, 1, 9, 1, 8, 5, 4, 2, 3, 2, 5, 2, 0, 0, 4, 6, 9, 5, 3, 5, 3, 0, 5, 2, 6, 8, 4, 8, 0, 1, 0, 2, 1, 0, 8, 6, 4, 6, 5, 7, 1, 9, 7, 4, 8, 8, 6, 9, 8, 8, 4, 1, 9, 5, 7, 5, 2, 5, 5, 7, 9, 2, 3, 8, 7, 6, 1, 4, 1, 1, 9, 0, 3, 5, 8, 1, 3, 8, 4, 3, 0, 1, 6, 0, 3, 2, 8, 0, 4, 6, 5, 8, 9, 4, 7, 6, 9, 5, 4, 8, 6, 3, 8, 7, 7, 1, 3, 2, 3, 2, 1, 0, 0, 3, 4, 4, 4, 6, 3, 9, 3, 9, 9, 8, 8, 2, 1, 7, 4, 8, 1, 0, 0, 8, 5, 8, 6, 1, 6, 8, 0, 1, 8, 1, 4, 1, 2, 0, 0, 6, 2, 5, 5, 3, 2, 6, 2, 3, 6, 3, 6, 9, 0, 1, 7, 1, 9, 8, 0, 1, 7, 2, 7, 1, 6, 8, 2, 1, 1, 7, 2, 6, 8, 0, 6, 9, 4, 3, 5, 6, 3, 2, 2, 9, 5]

    total_epochs, total_penalties = 0, 0
    training =False  # indicate this is not a training phase, no update is needed in Q-table

    list_pathlength = []
    list_episodetime = []
    episode=0
    for episode in range(numberofepisodes):
        start = time.time()
        path, pathedge="", ""
        current_pos = initialstate[episode]
        startpos = current_pos
        print('episode = ', episode, '  initial pos = ', current_pos)
        #current_pos = np.random.randint(1, (G.number_of_nodes()+1))#np.random.randint(0, G.number_of_nodes())  # Python excludes the upper bound
        currentState, possibleactionlist = getState(G,current_pos, qtable, False)
        #print('Starting position =', current_pos, ',  state = ', currentState)
        path = path + str(current_pos)+' '
        pathedge = pathedge + str(current_pos) + ' '
        epochs, penalties, rewardval, edgecount = 0, 0, 0, 0

        while (isterminalState(current_pos)==False):
            action = getbestaction(qtable,currentState)
            #action = getapproximatebestaction(G, current_pos, qtable, currentState)
            #execute the action and observe outcome
            #print('action ==', action)
            next_state_pos, rewardval = executeAction(G, current_pos, action)
            nextState, possibleactionlist = getState(G, next_state_pos, qtable, False)
            #print('===  action chosen = ', action)
            #print('next_state_pos =', next_state_pos, '  reward val = ', rewardval)
            #print('next state = ', nextState, '  next possible action list= ', possibleactionlist)


            path = path+' '+str(current_pos) + ' '+ str(action)+' ' + str(next_state_pos)
            if (current_pos != next_state_pos):
                pathedge = pathedge + str(next_state_pos) + ' '
                edgecount =edgecount+1

            currentState  = nextState
            current_pos = next_state_pos

            if rewardval < 0:
                penalties += 1

            epochs += 1
            if (epochs ==100):
                print("debug point")
                break

        # store time and path length
        list_episodetime.append(time.time() - start)
        list_pathlength.append(edgecount)

       # print('Episode= ', episode, 'rewards = ', rewardval, 'edge length = ', edgecount, '  change of states = ', epochs)
        total_penalties += penalties
        total_epochs += epochs
        #print('final path', path)
        print('visited edges/room = ', pathedge)
        print('shortest path =', list(nx.all_shortest_paths(G, str(startpos), str(goalState), weight=None)) )
        #break

    print(f"Results after {episode} episodes:")
    print(f"Average path length per episode: ", np.average(list_pathlength))
    print(f"Average number of change of states per episode: {total_epochs / numberofepisodes}")
    print('Average time per episode:', np.average(list_episodetime))
    print(f"Average penalty per episode: {total_penalties / numberofepisodes}")

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
def read_csv_file(fileloc):
    print('reading csv file, opening')
    with open(fileloc, mode='r') as file:
        reader = csv.reader(file)
        linenum=0
        for row in reader:
            linenum=linenum+1
            # parsing each column of a row
            for col in row:
                print("%10s" % col)
            #print(row)
            if(linenum ==4):
                break
# ----------------------------------------------------------------------------------------------------------
# Main function -- Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #read_csv_file('results/RL_test_57_rooms_63_doors_LR.csv')
    #exit()
    #flag variables
    generategraph = False  # true : generate graph, false : load graph from file
    training = False   # true : run training session, false : load trained q-table
    createtestset = False  # true: create the test set indicating starting point for the agent, false :

    # define path of the output directory where the environment graph and trained Q-table is located
    qtableoutputpath = 'results/Qtable'
    graphpath = 'results/RL_test_57_rooms_63_doors.xml'
    shortestpath = 'results/allshortestpaths'

    # Get the environment, environment is represented as a graph, either generate it or load from file
    if (generategraph == True):
        #environment is a maze of n rooms. It is represented as a graph. Generate the graph
        G = {}
        G = nx.read_graphml(graphpath)
        print('Number of nodes = ', G.number_of_nodes())
        Firehazardlocation(G.number_of_nodes())

    if (generategraph == False): # graph already exists in a .pickle file, need to load it
        G = nx.DiGraph()
        G = loadGraph(graphpath)
        #allShortestPaths ={}
        #allShortestPaths = pickle.load(open(shortestpath+'.pickle', 'rb'))
        #print(allShortestPaths)
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


    exit()
    # Training phase : Q learning agent, a
    Q_table={}
    if (training == True): # run the training session
        print('Start training')
        epsilonGreedyAlgo = True
        numTrainingEpisodes = 10000
        start = time.time()
        Q_table = QlearningTraining(G,numTrainingEpisodes, epsilonGreedyAlgo)
        print('Training time: ', (time.time() - start))
        print('-----------------------------------------------------------------')
        #printQtable(Q_table)
        print(len(Q_table.keys()))
        #store Q-table (after training) in a pickle file
        f = open(qtableoutputpath+'.pickle', 'wb')
        pickle.dump(Q_table, f)
        f.close()

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
    ########################################################################

    print('-------------------------------------------------------------------------------')
    print('-----Testing Q learning agent ----------------------------------------')
    numberoftestepisode=2000
    # -----------------create initial random state set
    initialstate = []
    if(createtestset == True):
        #for i in range(numberoftestepisode):
        i=0
        while i < numberoftestepisode:
            pos = np.random.randint(1, (G.number_of_nodes()+1))
            if pos not in initialstate:
                #print('pos', pos , i , numberoftestepisode)
                initialstate.append(pos)
                i=i+1
        f1 = open('results/testsetStartingpoint.pickle', 'wb')
        pickle.dump(initialstate, f1)
        f1.close()

    #--load initial test set from memory
    if (createtestset == False):
        initialstate = pickle.load(open('results/testsetStartingpoint.pickle', 'rb'))
        print(initialstate)
        counts, bins = np.histogram(initialstate)
        plt.hist(bins[:-1], bins, weights=counts,histtype='bar')
        plt.show()
    #print('strongly connected nodes = ', nx.strongly_connected_components(G))
    #checkshortestpathlength(G, initialstate)
    # minor changes in the graph
    #G1 = {}
    #G1 = copy.deepcopy(G)
    #G1.remove_edge('1', '61')
    #G1.remove_edge('2', '58')
    #G1.remove_edge('22', '34')
    #print('if edge exists = ',  G1.has_edge('2', '58'))
    #('2', '58')
    #('22', '34')
    #print(G1.edges())
    print('start testing')
    TestAgent(G,Q_table,numberoftestepisode, initialstate)

    #for testing robustness - slightly changed graph from the one the agent is trained
    #TestAgent(G1, Q_table, numberoftestepisode, initialstate)
    print('Q table size = ', len(Q_table.keys()))
    print('set of initial random starting point : ', initialstate)