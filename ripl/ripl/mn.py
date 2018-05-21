"""Custom topologies for Mininet

author: Brandon Heller (brandonh@stanford.edu)

To use this file to run a RipL-specific topology on Mininet.  Example:

  sudo mn --custom ~/ripl/ripl/mn.py --topo ft,4
"""

import os
import sys
from ripl.dctopo import FatTreeTopo, JellyFishTop

topos = { 'ft': FatTreeTopo,
          'jelly': JellyFishTop}
#,
#          'vl2': VL2Topo,
#          'tree': TreeTopo }
