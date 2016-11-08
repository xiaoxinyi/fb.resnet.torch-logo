#!/usr/bin/env python

import socket
import glob
import sys
sys.path.append('./gen-py')
sys.path.insert(0, glob.glob('/root/thrift/lib/py/build/lib*')[0])

from feature_extractor import FeatureExtractor
from feature_extractor.ttypes import *
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.server import TNonblockingServer

from my_extract_features import extract_featrues


port = 6006
ip = '0.0.0.0'

class FeatureExtractorHandler:
    def extract(self, img_dir):
        try:
            # Convert unicode to ascci
            ret = extract_featrues(str(img_dir))
            # ret = 'hello'
            # print img_dir
            print ret
            return ret + 'from ' + str(port)
        except:
            print('error ...')



handler = FeatureExtractorHandler()
processor = FeatureExtractor.Processor(handler)

transport = TSocket.TServerSocket(ip , port)
tfactory = TTransport.TBufferedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()

# server = TServer.TThreadPoolServer(processor,transport,tfactory,pfactory)
server = TNonblockingServer.TNonblockingServer(processor, transport, pfactory, threads=1)
# server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)


# from twisted.internet import reactor
# from thrift.transport import TTwisted

# reactor.listenTCP(port, TTwisted.ThriftServerFactory(processor=processor,iprot_factory=pfactory))
# reactor.run()

print "Starting thrift server ..."
server.serve()
print "Done!"
