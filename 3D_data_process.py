import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('data/galvanostatic-3d-data.txt', skiprows=9, names=['X', 'Y', 'Z', 'C'], delim_whitespace=True)

xs = []
cs = []
xs = np.linspace(0,2,50)
for x in xs:
    c = data.iloc[(data['X']-x).abs().argsort()[:20]].C.mean()
    cs.append(c)
    # xs.append(x)

plt.plot(xs,cs, '+')