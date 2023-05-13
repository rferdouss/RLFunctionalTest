import numpy as np
import networkx as nx
import copy
import time
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from keras.models import load_model
import random
import keras.backend as K
from keras import layers
from collections import deque

goalState = 100   # identify the destination room
maxreward =100  # max reward for reaching the destination room
numOfFirehazard =200
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
        roomnum = np.random.randint(0, G.number_of_nodes())
        firehazardpos.append(roomnum)
    f1 = open('results/firehazardposition.pickle', 'wb')
    pickle.dump(firehazardpos, f1)
    f1.close()


# ----------------------------------------------------------------------------------------------------------
def loadGraph(graphoutputpath):
    filename = graphoutputpath+'.pickle'
    G = pickle.load(open(filename, 'rb'))

    # adding doors, buttons, firehazard in rooms
    for n in G.nodes:
        G.nodes[n]['door'] = {g: 0 for g in G.out_edges(n)}  # 0 means door is closed
        G.nodes[n]['button'] = {g: 0 for g in G.out_edges(n)}   # 0 means button is not pressed
        G.nodes[n]['name'] = 'room'+str(n)
        G.nodes[n]['firehazard'] = 0  # initially no firehazard

    # assigning fire hazard in  rooms
    firehz = pickle.load(open('results/firehazardposition.pickle', 'rb'))
    for i in range(len(firehz)):
        #roomnum = np.random.randint(0, G.number_of_nodes())
        G.nodes[firehz[i]]['firehazard'] = 1  # 1 means firehazard exists inside that room
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
    if currentposition == goalState:
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
        #print('max val(state, action)', qtable[state])
        max_key = max(qtable[state], key=qtable[state].get)
        bestaction = max_key
    #print('best action ', bestaction)
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
def getState(G,node_pos):
    state =""
    possibleActionList =[]
    # get the number of doors possible to go from this room
    for key in G.nodes[node_pos]['door'].keys():
        # d means door, following current and connecting room position, and its status (open -1, closed-0).Example : d-2-3-0 (i.e., it is a door from room 2 to room 3. The status of the door is closed)
        doorname =  'd-'+ str(node_pos)+'-'+str(key[1])
        state = state +" "+doorname+'-'+str(G.nodes[node_pos]['door'][key])
        possibleActionList.append(doorname)

    for key in G.nodes[node_pos]['button'].keys():
        # b means door, following current and connecting room position, and its status (pressed -1, not pressed-0).Example : b-2-3-0 (i.e., it is a button for a door from room 2 to room 3. The status of the button is not pressed)
        buttonname = 'b-' + str(node_pos)+'-'+str(key[1])
        state = state + " " + buttonname+'-'+str(G.nodes[node_pos]['button'][key])
        possibleActionList.append(buttonname)

    if (G.nodes[node_pos]['firehazard']== 1):
        # firehazard is an element inside a room. so the presentation is fz means firehazard, follows by the room number and again the room number, then always 1 indicating it exists
        firehazard = 'fz-' + str(node_pos)+'-'+str(node_pos)
        state = state + " " + firehazard+'-1'
        possibleActionList.append(firehazard)

    state =state.lstrip()
    #if (training == True):  # training time
    #    qtable = setqtableentry(G, node_pos, state, qtable, possibleActionList)
    return state, possibleActionList

# ----------------------------------------------------------------------------------------------------------
def executeAction(G,pos,action):
    nextstatepos =-1
    rewardval =-1
    x = action.split("-")
    #print('split', x[0], x[1],x[2])
    key1 =int(x[1])
    key2 =int(x[2])
    if (x[0] =='d'):
        #print('it is door, value ', G.nodes[pos]['door'][(key1, key2)])
        if (G.nodes[pos]['door'][(key1,key2)]==1):  # if the door is open then we can go to the connecting door
            nextstatepos = key2  # next room position
            if (key2 == goalState):
                rewardval = maxreward # get the reward from going to the connecting room
            else:
                rewardval = 0
            #print('Door is open, going to next room ', G.nodes[pos]['door'][(key1, key2)])
        if (G.nodes[pos]['door'][(key1,key2)]==0):  # door is closed, nothing to be done. First need to press the button that opens this door
            nextstatepos = key1  # staying in the current room
            rewardval = 0  #NEED TO CHECK
            #print('Door is closed, first need to press the button that opens it ', G.nodes[pos]['door'][(key1, key2)])

    if (x[0] == 'b'):
        #print('it is button, value ', G.nodes[pos]['button'][(key1, key2)])
        if (G.nodes[pos]['button'][(key1,key2)]==1):  # if the button is already pressed, do noting
            #print('button value = ', G.nodes[pos]['button'][(key1,key2)], 'already pressed')
            G.nodes[pos]['door'][(key1, key2)] = 1  # refresh the status of the door, it should be open
            nextstatepos = key1  # next room position
            rewardval = 0
        if (G.nodes[pos]['button'][(key1,key2)]==0):  # if the button is not pressed, press the button to open the connecting door
            #print('button value = ', G.nodes[pos]['button'][(key1, key2)], 'Not pressed')
            nextstatepos = key1  # staying in the current room
            rewardval = 0
            # change the button state as pressed, do does the door state as open
            G.nodes[pos]['button'][(key1, key2)] = 1  # button is pressed
            G.nodes[pos]['door'][(key1, key2)] = 1  # door is open
            #print('Changing button value ', G.nodes[pos]['button'][(key1, key2)])

    if (x[0] == 'fz'):
        nextstatepos = key1
        rewardval = -10000   #PENALTY IF INTERACT WITH FIREHAZARD
        #print('it is firehazard, value ', G.nodes[pos]['firehazard'])

    return nextstatepos, rewardval
# ----------------------------------------------------------------------------------------------------------
def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

# ----------------------------------------------------------------------------------------------------------
def train(replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

# ----------------------------------------------------------------------------------------------------------
def DQN(G, train_episodes):
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    stateSpace = 300 #G.number_of_nodes + G.number_of_edges() * 2 + G.number_of_edges() * 2
    actionSpace = 30
    model = agent(stateSpace, actionSpace)
    # Target Model (updated every 100 steps)
    target_model = agent(stateSpace, actionSpace)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        G_ = copy.deepcopy(G)
        total_training_rewards = 0
        # Pick up a state randomly as a starting state for the agent in a learning episode
        possibleActionList = []
        current_pos = np.random.randint(0, G_.number_of_nodes())  # Python excludes the upper bound
        currentState, possibleActionList = getState(G_, current_pos)
        print('initial state = ', currentState, 'current position = ', current_pos)
        #done = False
        while (isterminalState(current_pos)==False):
            steps_to_update_target_model += 1
            #if True:
            #    env.render()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = random.choice((possibleActionList))
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = currentState
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)

            next_state_pos, rewardval = executeAction(G_, current_pos, action)
            nextState, possibleActionList = getState(G_, next_state_pos)
            print('action  = ', action, ' next state  = ', next_state_pos)
            #new_observation, reward, done, info = env.step(action)
            replay_memory.append([currentState, action, rewardval, nextState, isterminalState(current_pos)])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or isterminalState(current_pos):
                train(replay_memory, model, target_model, isterminalState(current_pos))

            currentState = nextState
            current_pos = next_state_pos
            total_training_rewards += rewardval


            if isterminalState(current_pos):
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

# ----------------------------------------------------------------------------------------------------------
# Main function -- Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #flag variables
    generategraph = False  # true : generate graph, false : load graph from file
    training = True   # true : run training session, false : load trained q-table

    # define path of the output directory where the environment graph and trained Q-table is located
    qtableoutputpath = 'results/Qtabledqn'
    graphoutputpath = 'results/graphdqn'

    # Get the environment, environment is represented as a graph, either generate it or load from file
    if (generategraph == True):
        #environment is a maze of n rooms. It is represented as a graph. Generate the graph
        num_node = 10
        edge_probability =0.4#0.01
        GenerateGraph(num_node, edge_probability, graphoutputpath)

    if (generategraph == False): # graph already exists in a .pickle file, need to load it
        G = nx.DiGraph()
        G = loadGraph(graphoutputpath)
        print('number of edges = ', G.number_of_edges())
        #print(G.nodes(data =True))
        #print('nodes ',G.number_of_nodes())
        #print(nx.degree_histogram(G))
        #print(nx.degree(G))
        #plot_graph(G)

    #Training with DeepQNetwork
    if (training==True):
        train_episodes =1
        DQN(G, train_episodes)

    if (training ==False):
        print('load trained neural network')