from mininet.node import Controller
import os

POXDIR = os.getcwd() + '/../..'

class JELLYPOX( Controller ):
    def __init__( self, name, cdir=POXDIR,
                  command='python pox.py', cargs=('log --file=jelly.log,w openflow.of_01 --port=%s pox.forwarding.topo_proactive' ),
                  **kwargs ):
        Controller.__init__( self, name, cdir=cdir,
                             command=command,
                             cargs=cargs, **kwargs )
controllers={ 'jelly': JELLYPOX }
