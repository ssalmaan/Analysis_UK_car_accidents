import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import KMeans


def run_dbscan(X, param, i):
    plt.figure(i)

    bbox = [49.5, 58.3, -6.0, 2.0]
    m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
                llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')
    m.drawcoastlines()
    m.fillcontinents(color='white',lake_color='blue')
    m.drawmapboundary(fill_color='dodgerblue')
    
    db = DBSCAN(eps=param[0], min_samples=param[1]).fit(X)
    # db = KMeans(n_clusters=66, random_state=0)
    # db = DBSCAN(eps=0.1, min_samples=5).fit(X)
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print(param)
    print('Number of clusters: %d' % n_clusters_)
    print('Number of noise points: %d' % n_noise_)
    
    # print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))
    
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        x,y = m(xy[:, 0], xy[:, 1])
        m.plot(x,y, 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
        xy = X[class_member_mask & ~core_samples_mask]
        x,y = m(xy[:, 0], xy[:, 1])
        m.plot(x,y, 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=0.3)
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=2)
    
    plt.title('Min-points={0}, {1} clusters'.format(param[1], n_clusters_))
    plt.show()



data = pd.read_csv("accidents_2012_to_2014.csv")
data = data[data['Accident_Severity'] == 1]
data = data[['Longitude', 'Latitude']]

X=data.values

params = [[0.1, 5], [0.1, 10], [0.1, 25]]
params = [[0.1, minpoints] for minpoints in [1, 3, 6, 10, 15, 20]]
params = [[0.1, 25]]
for i, param in enumerate(params):
    run_dbscan(X, param, i)


"""

bbox = [49.5, 58.3, -6.0, 2.0]
m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
            llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')
m.drawcoastlines()
m.fillcontinents(color='white',lake_color='blue')
m.drawmapboundary(fill_color='dodgerblue')


db = KMeans(n_clusters=66, random_state=0).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

labels = db.labels_


unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]# & core_samples_mask]
    x,y = m(xy[:, 0], xy[:, 1])
    m.plot(x,y, 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)


plt.title('K-Means, 66 clusters')
plt.show()
"""
