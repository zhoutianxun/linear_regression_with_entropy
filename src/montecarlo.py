import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def random_walk(model, step_size=1e-3):
    noise = np.random.normal(size=model.shape) * step_size
    model += noise
    return model

def test_model(model, X_test, y_test):
    y_pred = X_test @ model
    return mean_squared_error(y_test, y_pred)

def stopping_criteria(h, thresh=0.8):
    m = np.mean(h)
    return np.all(h >= thresh * m)

def exponential_moving_average(past_ave, new_value, i, factor=10):
    return past_ave + (new_value - past_ave) / min(i, factor)

def stopping_criteria_saturation(h, delta_h_prev, thresh=100):
    delta_h = np.sum(h - np.min(h))
    return np.abs(delta_h - delta_h_prev) < thresh

def wang_landau(model, x, y, f=np.exp(1), init_g=None, step_size=2e-3, low=0, high=1, bins=100):
    bin_width = (high-low)/bins
    b = np.linspace(low, high, bins, endpoint=False)
    h = np.zeros_like(b)
    if init_g is not None:
        g = init_g
    else:
        g = np.zeros_like(b) + np.log(f)
    
    E1 = test_model(model, x, y)
    i = 1
    delta_h_prev = 0
    check_saturation_every = 1000
    while True:
        if i % check_saturation_every == 0:
            if stopping_criteria_saturation(h, delta_h_prev, int(check_saturation_every*0.05)):
                plt.figure(figsize=(2,1))
                plt.bar(b, h)
                plt.show()
                break
            else:
                delta_h_prev = exponential_moving_average(delta_h_prev, np.sum(h - np.min(h)), i//1000)

        E2 = high
        while E2 >= high or E2 < low:
            model_2 = random_walk(model.copy(), step_size=step_size)
            E2 = test_model(model_2, x, y)
        g1 = g[int((E1 - low)/bin_width)]
        g2 = g[int((E2 - low)/bin_width)]
        try:
            prob = min(np.exp(g1 - g2), 1)
        except RuntimeWarning:
            print(f'Overflow, g1: {g1}, g2: {g2}')
            return b, g, h

        if np.random.random() < prob:
            model = model_2
            g[int((E2 - low)/bin_width)] += np.log(f)
            h[int((E2 - low)/bin_width)] += 1
            #A[int((E1 - low)/bin_width)] += 1
            E1 = E2 
        else:
            g[int((E1 - low)/bin_width)] += np.log(f)
            h[int((E1 - low)/bin_width)] += 1
            #R[int((E1 - low)/bin_width)] += 1 

        if (i+1)%100000==0:
            print(np.mean(h)*0.8, np.min(h))

        i += 1
    return b, g, h