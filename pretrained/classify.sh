#!/usr/bin/env zsh
find ../adidas -iname "*.jpg" | xargs th classify.lua ../checkpoints-50/model_best.t7 | tee  adidas.log
