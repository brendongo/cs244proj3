import os
import sys
import numpy as np

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.node import OVSController
from mininet.node import Controller
from mininet.node import RemoteController
from mininet.cli import CLI
from jelly_pox import JELLYPOX
from subprocess import Popen
from ripl.ripl.dctopo import JellyFishTop
from time import sleep, time
from tqdm import tqdm

import re
from time import sleep, time
from sys import exit, stdout, stderr
from optparse import OptionParser
from json import dumps

from mininet.net import Mininet
from mininet.node import Controller, CPULimitedHost
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, warn, output

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Host
from mininet.util import quietRun, natural, custom

from decimal import Decimal

# Adapted from
# https://github.com/mininet/mininet-tests/edit/master/pairs/pair_intervals.py

def listening( src, dest, port=5001 ):
    "Return True if we can connect from src to dest on port"
    cmd = 'echo A | telnet -e A %s %s' % (dest.IP(), port)
    result = src.cmd( cmd )
    return 'Connected' in result

# Iperf pair test
def info(*args):
    print args

def iperfPairs( clients, servers, experimentName, num_flows):
    "Run iperf semi-simultaneously one way for all pairs"
    pairs = len( clients )
    plist = zip( clients, servers )

    info( '*** Clients: %s\n' %  ' '.join( [ c.name for c in clients ] ) )
    info( '*** Servers: %s\n' %  ' '.join( [ c.name for c in servers ] ) )
    info( "*** Shutting down old iperfs\n")
    quietRun( "pkill -9 iperf" )
    info( "*** Starting iperf servers\n" )
    for dest in servers:
        dest.cmd( "iperf -s -p 5555&" )
    info( "*** Waiting for servers to start listening\n" )
    for src, dest in plist:
        info( dest.name, '' )
        while not listening( src, dest, 5555 ):
            print  '.'
            sleep( .5 )
    info( '\n' )
    info( "*** Starting iperf clients\n" )
    for src, dest in plist:
        output_file = "results/iperf_{}_{}_{}_{}".format(experimentName, num_flows, src.name, dest.name)
        src.sendCmd( "sleep 1; iperf -l %fM -f M -p 5555 -t %s -i .5 -P %d -c %s > %s" % (
            1/float(num_flows), 10, num_flows, dest.IP(), output_file))

        #src.sendCmd( "sleep 1; iperf -f M -p 5555 -t %s -i .5 -c %s > %s" % (
        #    10, dest.IP(), output_file) )
    info( "*** Waiting for clients to complete\n" )
    results = []
    for src, dest in plist:
        result = src.waitOutput()
        print result
        results.append(result)
    
    output_file = open(experimentName, "w")
    for result in results:
        output_file.write(result)


def experiment(net, experiment_name, num_flows):
    for h in net.hosts:
        print "disable ipv6"
        h.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
        h.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1")
        h.cmd("sysctl -w net.ipv6.conf.lo.disable_ipv6=1")

    for sw in net.switches:
        print "disable ipv6"
        sw.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
        sw.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1")
        sw.cmd("sysctl -w net.ipv6.conf.lo.disable_ipv6=1")
    
    net.start()
    sleep(3)
    #net.pingAll()

    permutation = net.hosts
    np.random.shuffle(permutation)
    srcs = []
    dsts = []
    for i in xrange(len(permutation)):
        srcs.append(permutation[i])
        dsts.append(permutation[(i + 1) % len(net.hosts)])
    iperfPairs(srcs, dsts, experiment_name, num_flows)
    net.stop()

def main():
    experiment_name = sys.argv[1]
    num_flows = int(sys.argv[2])
    
    topo = JellyFishTop()
    net = Mininet(
            topo=topo, host=CPULimitedHost, link=TCLink,
            controller=lambda name: RemoteController(name, port=6633))
    experiment(net, experiment_name, num_flows)

if __name__ == "__main__":
    main()
