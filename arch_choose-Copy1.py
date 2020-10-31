import pandas as pd

import numpy as np

import pickle

import os

import time

import lwmw

cycle = {"/gpu:1" : "/gpu:2", "/gpu:2" : "/gpu:3", "/gpu:3": "/gpu:1"}

metrics = pd.read_csv("metrics100.csv")

settings = {
    "outs" : 5,
    "input_len" : len(metrics),
    "architecture" : [11,6],
    "inputs" : len(metrics.columns)-5,
    "activation" : "sigmoid",
    "gpu_name": "/gpu:1",
    "i": 0
}

with open('/home/gpu/lab/medical_research/arch_dump/settings.pickle', 'wb') as f:
    pickle.dump(settings, f)

for i in range(0,4):
    os.system("python /home/gpu/lab/medical_research/step_arch_choose.py")
    print("*" * 5 + f" STEP {i} COMPLETE!" + "*" * 5)
    settings["i"] = i
    settings["gpu_name"] = cycle[settings["gpu_name"]]
    with open('/home/gpu/lab/medical_research/arch_dump/settings.pickle', 'wb') as f:
        pickle.dump(settings, f)
    
print("Complite")