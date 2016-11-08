#!/usr/bin/env python

import socket
import glob
import sys
sys.path.append('./gen-py')
sys.path.insert(0, glob.glob('/root/thrift/lib/py/build/lib*')[0])
from feature_extractor import FeatureExtractor

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

port = 6006

try:

    transport = TSocket.TSocket('localhost', port)

    # transport = TTransport.TBufferedTransport(transport)
    # for nonblockserver
    transport = TTransport.TFramedTransport(transport)

    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    client = FeatureExtractor.Client(protocol)
    transport.open()

    # model_path = 'models-torch/resnet-18.t7'
    image_dir = 'data/pradalogo'

    msg = client.extract(image_dir)
    print msg

    transport.close()
except Thrift.TException, ex:
    print "%s" % (ex.message)
