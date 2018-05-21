import os
import re

def compile_results(routing, congestion):
    experiment = routing + '_' + congestion
    throughputRegex = '(\d+\.?\d*) *MBytes/sec'

    throughputs = []
    for filename in os.listdir('results/'):
        if experiment in filename:
            with open('results/' + filename, 'r') as file:
                file_str = file.readlines()[-1]
                match = re.search(throughputRegex, file_str)
                throughputs.append(float(match.group(1)))

    averageThroughput = sum(throughputs) / float(len(throughputs))
    return averageThroughput

def main():
    routing_options = ["ecmp", "shortest_paths"]
    congestion_options = ['1', '8']

    for routing in routing_options:
        for congestion in congestion_options:
            throughput = compile_results(routing, congestion)
            print routing, " TCP " + congestion + " flow(s):", throughput, " MBytes/sec || as a fraction of maximum: ", throughput * 8 / 10.0 
            print "----"

if __name__ == '__main__':
    main()
