import networkx as nx
import matplotlib.pyplot as plt
from random import randrange
import numpy as np

# Find the total graph weight

def find_total_weight(G):
    tw = 0
    for edge in G.edges():
        tw += G.get_edge_data(edge[0], edge[1], "weight")['weight']
    return tw


#Spanner Algorithm

def SPANNER(graph, stretch_factor):
    sorted_edges_of_g = sorted(graph.edges(data=True), key=lambda t: t[2].get('weight', 1))
    H = nx.Graph()
    H.add_nodes_from(graph)
    # for edge {u, v} edge[0] = u and edge[1] = v
    for edge in sorted_edges_of_g:
        edge_weight = graph.get_edge_data(edge[0], edge[1], "weight")['weight']
        if nx.has_path(H, edge[0], edge[1]):
            sh_path_len = nx.dijkstra_path_length(H, edge[0], edge[1], weight='weight')
            if stretch_factor * edge_weight < sh_path_len:
                H.add_edge(edge[0], edge[1], weight=edge_weight)
        else:
            H.add_edge(edge[0], edge[1], weight=edge_weight)

    return H
#Make random connected graph

def random_connected_graph(number_of_vertices, probability):
    grp = nx.erdos_renyi_graph(number_of_vertices, probability)
    while not nx.is_connected(grp):
        grp = nx.erdos_renyi_graph(number_of_vertices, probability)
    return grp
#Make random connected graph with different weights

def random_connected_graph_with_different_weights(number_of_vertices, probability):
    grp = random_connected_graph(number_of_vertices, probability)
    for e in grp.edges():
        # randrange give us some random number between 0 to 9
        grp[e[0]][e[1]]['weight'] = (randrange(50) + 1)

    return grp

#Check the number of edges in the graph.

def check_num_of_edges(number_of_vertices, probability, num_of_graphs):
    # we will test 100 times the num of edges
    if probability == 0:
        print(0)
    sum = 0
    for x in range(num_of_graphs):
        grp = random_connected_graph(number_of_vertices, probability)
        for e in grp.edges():
            grp[e[0]][e[1]]['weight'] = 1

        H = (SPANNER(grp, 1))
        sum += H.number_of_edges()

    # print(sum / num_of_graphs)
#Experiment 1

def exp1(number_of_vertices, probability, num_of_graphs):
    xis = []
    yis = []
    for x in range(num_of_graphs):
        grp = random_connected_graph(number_of_vertices, probability)
        for e in grp.edges():
            grp[e[0]][e[1]]['weight'] = 1

        # print("before spanner function: ", grp.number_of_edges())
        xis.append(grp.number_of_edges())
        H = SPANNER(grp, 1)
        # print("after spanner function: ", H.number_of_edges())
        yis.append(H.number_of_edges())

    plt.xlabel('Graph-Num Of Edges')
    plt.ylabel('Graph-Num Of Edges after spanner')
    plt.plot(xis, yis)
    plt.scatter(xis, yis)
    plt.savefig('exp1.pdf')
    plt.show()


# exp1(30, 0.5, 150)

#Experiment 2
#num of vers
def exp2(number_of_vertices, probability, num_of_graphs):
    plt.xlabel('stretch factor')
    plt.ylabel('number of edges')
    for x in range(num_of_graphs):
        xis = []
        yis = []
        grp = random_connected_graph(number_of_vertices, probability)
        for e in grp.edges():
            grp[e[0]][e[1]]['weight'] = 1
        for stretch_fact in range(1, 10, 2):
            xis.append(stretch_fact)
            yis.append(SPANNER(grp, stretch_fact).number_of_edges())
        plt.plot(xis, yis)

    plt.savefig('exp2.pdf')
    plt.show()


# exp2(30, 0.5, 100)


def find_min_weight_path(G):
    weights = []
    for e in G.edges():
        weights.append(nx.dijkstra_path_length(G, e[0], e[1], weight='weight'))
    return min(weights)

#Experiment 3

def exp3(number_of_vertices, probability, num_of_graphs):
    xis = []
    yis = []
    for x in range(num_of_graphs):
        grp = random_connected_graph_with_different_weights(number_of_vertices, probability)

        # print("before spanner function: ", grp.number_of_edges())
        xis.append(grp.number_of_edges())
        x = grp.number_of_edges()
        H = SPANNER(grp, 1)
        # print("after spanner function: ", H.number_of_edges())
        yis.append(H.number_of_edges())
        y = H.number_of_edges()

        plt.xlabel('Graph-Num Of Edges')
        plt.ylabel('Graph-Num Of Edges after spanner')
    plt.scatter(xis, yis)
    plt.savefig('exp5.pdf')
    plt.show()


# exp3(40, 0.5, 100)

#Experiment 4
#100 ver at leats and different denisites
def exp4(number_of_vertices, probability, num_of_graphs):
    xis = []
    yis = []
    for x in range(num_of_graphs):
        grp = random_connected_graph_with_different_weights(number_of_vertices, probability)
        xis.append(find_total_weight(grp))
        yis.append(find_total_weight(SPANNER(grp, 1)))

    plt.xlabel('Graph - Weight')
    plt.ylabel('Spanner - Graph - Weight')
    labels = ['G1', 'G2', 'G3', 'G4', 'G5',
              'G6', 'G7', 'G8', 'G9', 'G10'
        , 'G11', 'G12', 'G13', 'G14', 'G15'
        , 'G16', 'G17', 'G18', 'G19', 'G20']
    x = np.arange(len(labels))  # the label locations
    width = 0.45  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, xis, width, label='Original Graph')
    rects2 = ax.bar(x + width / 2, yis, width, label='Spanner Graph')
    ax.set_ylabel('Weight')
    ax.set_title('Graph and Spanner Graph Weight')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('exp4.pdf')
    plt.show()

# exp4(100, 0.5, 20)


def exp4second(number_of_vertices, num_of_graphs):
    plt.xlabel('Density')
    plt.ylabel('Weight')
    # change to False in order to test original graph
    isspanner = True
    xis = []
    yis = []
    yissp = []
    for probability in np.arange(0.1, 1, 0.1):
        for x in range(num_of_graphs):
            grp = random_connected_graph_with_different_weights(number_of_vertices, probability)
            xis.append(probability)
            print("probability: ", probability)
            yis.append(find_total_weight(grp))
            if isspanner:
                yissp.append(find_total_weight(SPANNER(grp, 1)))
                print("total weight: ", find_total_weight(SPANNER(grp, 1)))
    if not isspanner:
        plt.plot(np.unique(xis), np.poly1d(np.polyfit(xis, yis, 1))(np.unique(xis)))
    else:
        plt.plot(np.unique(xis), np.poly1d(np.polyfit(xis, yissp, 3))(np.unique(xis)))
    plt.title('Original Graph')
    plt.savefig('exp4_density_changed_original.pdf')
    plt.show()

# exp4second(100, 20)

#Experiment 5
# number of vertices
def exp5(number_of_vertices, probability, num_of_graphs):
    for x in range(num_of_graphs):
        xis = []
        yis = []
        grp = random_connected_graph_with_different_weights(number_of_vertices, probability)
        # xis.append(find_total_weight(grp))
        # yis.append(find_total_weight(SPANNER(grp, 1)))
        for stretch_fact in range(1, 8, 2):
            xis.append(stretch_fact)
            yis.append(find_total_weight(SPANNER(grp, stretch_fact)))

        plt.xlabel('Stretch Factor')
        plt.ylabel('Total Graph Weight')
        plt.plot(xis, yis)
    plt.savefig('exp5.pdf')
    plt.show()


# exp5(100, 0.5, 20)

#Experiment 6
# stretch factor = 3
def exp6(number_of_vertices, num_of_graphs):
    plt.xlabel('Density')
    plt.ylabel('Number of edges')
    xis = []
    yisspaner = []
    # yis = []
    for probability in np.arange(0.1, 1, 0.1):
        for x in range(num_of_graphs):

            grp = random_connected_graph_with_different_weights(number_of_vertices, probability)
            xis.append(probability)
            # yis.append((grp.number_of_edges()))
            yisspaner.append((SPANNER(grp, 5)).number_of_edges())

    plt.plot(np.unique(xis), np.poly1d(np.polyfit(xis, yisspaner, 3))(np.unique(xis)))
    plt.title('Spanner Graph')
    plt.savefig('exp6spanner.pdf')
    plt.show()

# exp6(100, 20)
    

