
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(precision=4, suppress=True)

from sklearn.datasets import make_blobs


x, y = make_blobs(n_samples=100, centers=4,
                  random_state=500, cluster_std=1.25)


model = KMeans(n_clusters=4, random_state=0)
model.fit(x)

y_ = model.predict(x)
print(y_)

plt.figure(figsize=(10, 6))
plt.scatter(x[:, 0], x[:, 1], c=y_,
            cmap='coolwarm')
plt.show()
