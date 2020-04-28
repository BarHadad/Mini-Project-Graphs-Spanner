# Mini-Project-Graphs-Spanner
The algorithm was taken from the paper "On sparse spanners of weighted graphs" by I. Alth\"ofer, G. Das, D. Dobkin, D. Joseph, and J. Soares. 
The algorithm gets a graph G = (V, E) and a stretch factor T and returns a T-spanner G'=(V,E') for G. 
Spanner definition: 
Let G = (V, E) be a connected n-vertex graph with arbitrary positive edge weights. A sub-graph G' = (V, E') is a T-spanner of G if between each pair of vertices v1, v2 the distance between v1 and v2 in G' is at most T-times longer than the distance in G. T is called the stretch factor associated with T. 
