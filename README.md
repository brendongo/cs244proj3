# Install

`./install.sh`
`export PYTHONPATH=root of this repo`

# Generate Figure 9
`python main.py`

# Generate Table 1:
`./run.sh`


# Progress

We have implemented the RRG generation method described in the paper,
consisting of two phases:

    - While possible, randomly select two nodes that aren't connected who both
      have free ports and connect them.
    - While there is any node u with more than two free ports left, randomly
      select an edge (v, w). Disconnect (v, w) and connect (u, v), (v, w).

This results in an RRG, where every node is guaranteed to have at most 1 free
port.

We further implemented Yen's Loopless K-shortest Paths algorithm to obtain the
k-shortest paths between any two nodes in the generated RRG. We generate 
random permutation traffic and count the paths through the edges in the graph,
reproducing Figure 9 from the paper.

# Reproduce

