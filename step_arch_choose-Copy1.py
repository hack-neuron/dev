import lwmw
import pickle
import pandas as pd
import numpy as np

with open('/home/gpu/lab/medical_research/arch_dump/settings.pickle', 'rb') as f:
    settings = pickle.load(f)
    
with open('/home/gpu/lab/medical_research/arch_dump/many_hist.pickle', 'rb') as f:
    hist_many = pickle.load(f)
    
with open('/home/gpu/lab/medical_research/arch_dump/p_many.pickle', 'rb') as f:
    p_many = pickle.load(f)

metrics = pd.read_csv("metrics100.csv")
    

hist_many[settings["i"]], p_many[settings["i"]] = lwmw.levmarq(
    settings, 
    x_train=metrics.values[:,:-5], 
    y_train=metrics.values[:,-5:], 
    mu_init=5.0, 
    min_error=8.33e-4, 
    max_steps=500, 
    mu_multiply=10, 
    mu_divide=10, 
    m_into_epoch=10, 
    verbose=True
)

with open('/home/gpu/lab/medical_research/arch_dump/many_hist.pickle', 'wb') as f:
    pickle.dump(hist_many, f)
    
with open('/home/gpu/lab/medical_research/arch_dump/p_many.pickle', 'wb') as f:
    pickle.dump(p_many, f)