#!/bin/bash

rm -f results/*

python pox/pox.py riplpox.riplpox --topo=jelly  --routing=ecmp --mode=reactive > /dev/null 2>&1 &
sudo mn -c && sudo python build_topology.py ecmp 1
pkill python

python pox/pox.py riplpox.riplpox --topo=jelly  --routing=ecmp --mode=reactive &
sudo mn -c && sudo python build_topology.py ecmp 8
pkill python

python pox/pox.py riplpox.riplpox --topo=jelly  --routing=shortest_paths --mode=reactive &
sudo mn -c && sudo python build_topology.py shortest_paths 1
pkill python

python pox/pox.py riplpox.riplpox --topo=jelly  --routing=shortest_paths --mode=reactive &
sudo mn -c && sudo python build_topology.py shortest_paths 8
pkill python

python generate_table1.py > table1.txt
cat table1.txt
