import lwmw
import pickle
import pandas as pd
import numpy as np

metrics = pd.read_csv("data_1.csv")
metrics = metrics.drop(columns="name")
metrics = metrics.append(pd.read_csv("data_2.csv").drop(columns="name")).append(pd.read_csv("data_3.csv").drop(columns="name"))

settings = {
        "outs" : 5,
        "input_len" : len(metrics),
        "architecture" : [20,8],
        "inputs" : len(metrics.columns) - 5,
        "activation" : "sigmoid"
    }

hist_many, p_many = lwmw.levmarq(
    settings, 
    x_train=metrics.values[:,:-5], 
    y_train=metrics.values[:,-5:], 
    mu_init=10.0, 
    min_error=1e-4, 
    max_steps=5000, 
    mu_multiply=10, 
    mu_divide=10, 
    m_into_epoch=10, 
    verbose=True
)

np.save("hist_20_8", hist_many)
np.save("p_20_8", p_many)