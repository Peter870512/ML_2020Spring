import numpy as np

def schedule_sampling(step_cnt, step):
    total_step = 12000
    summary_step = 300
    curr_step = step_cnt * summary_step + step
    Type = 'inv'
    if Type == 'linear': # linear decay
        sampling = -1 * curr_step / total_step + 1
    elif Type == 'exp': # exponential decay
        c = 0.0005
        sampling = np.exp(-curr_step * c)
    elif Type == 'inv': # inverse sigmoid decay
        sampling = 1 / (1 + pow(0.5, (-(curr_step - 6000) / 700))) # 1 ./ (1 + 0.5 .^ (-(x-6000)/700))
    return sampling