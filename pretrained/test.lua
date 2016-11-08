fe = require('pretrained/my-extract-features')

local debugger = require('fb.debugger')
debugger.enter()

a = fe('../models-torch/resnet-18.t7')
