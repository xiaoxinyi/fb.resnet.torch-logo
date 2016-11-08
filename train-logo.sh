#!/usr/bin/env zsh
th main.lua -retrain models-torch/resnet-152.t7 -data logo-data  -resetClassifier true -nClasses 45  -LR 0.001 -save checkpoints-152 -batchSize 16 -tenCrop true | tee checkpoints-152/log.txt

# th main.lua  -data logo-data  -resume checkpoints-50  -tenCrop true -nEpochs 400 -epochNumber 41 -save checkpoints-50 -batchSize 16
