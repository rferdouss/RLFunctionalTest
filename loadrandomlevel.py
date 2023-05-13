import networkx as nx
#import matplotlib as plt
import matplotlib.pyplot as plt

def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G,pos)
    plt.show()


if __name__ == '__main__':
    inputgraph = 'results/RL_test_2000_rooms.xml'
    G = nx.read_graphml(inputgraph)
    print('Number of nodes= ', G.number_of_nodes(), '   number of edges = ',G.number_of_edges())
    #print(G.nodes(data=True))
    #for n in G.nodes:
    print('degree = ', G.degree())
    plot_graph(G)
    #nx.draw(G)
    print('is directed graph = ', nx.is_directed(G))
    print('edges from 100 ', G.edges('100'))
    print('edges = ',G.edges())
    print('is connected = ', nx.is_connected(G))
    print('nodes',  G.nodes())