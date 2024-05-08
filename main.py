from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
from vdbscan import VDBSCAN
from Data_make import *
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

scaler = MinMaxScaler()

# fetch dataset
dry_bean = fetch_ucirepo(id=602)

# data (as pandas dataframes)
X = dry_bean.data.features
y = dry_bean.data.targets

X = pd.DataFrame(scaler.fit_transform(X) * 10, columns=X.columns)

cov_matrix = X.cov()


np.fill_diagonal(cov_matrix.values, np.nan)

# Finding the indices of the maximum value in the covariance matrix
max_cov_index = np.unravel_index(np.nanargmax(np.abs(cov_matrix)), cov_matrix.shape)

# Getting the names of the columns corresponding to the maximum covariance
feature1, feature2 = cov_matrix.columns[max_cov_index[0]], cov_matrix.columns[max_cov_index[1]]
max_cov_value = cov_matrix.iloc[max_cov_index[0], max_cov_index[1]]

print(f"The pair of attributes with the highest covariance is {feature1} and {feature2} with a covariance of {max_cov_value}.")


np.set_printoptions(threshold=np.inf)
X_new = np.array(X[["Area", "Perimeter"]])

print(X_new.shape)

#create data
data = []
for i in range(5):
    data.append(create_rand_blob(3))
    data.append(create_rand_moon())
    data.append(create_rand_circle())
    data.append(create_rand_diff_blob())
    data.append(create_rand_diff_moon())

# i = 0
# for ds in data:
#     ds = np.array(ds)
#     # vdb = VDBSCAN(kappa= .05)
#     # labels_1 = vdb.fit(ds, eta=0.5).labels_
#     HDB = HDBSCAN(min_samples=10)
#     labels_2 = HDB.fit(ds).labels_
#     # labels_3 = DBSCAN(min_samples=5, eps=.5).fit(ds).labels_
#     # plt.scatter(ds[:, 0], ds[:, 1], c=labels_1)
#     # plt.title("vdb")
#     # plt.savefig("vdb"+str(i)+".png")
#     # plt.close()
#     plt.scatter(ds[:, 0], ds[:, 1], c=labels_2)
#     plt.title("HDB")
#     plt.savefig("HDB"+str(i)+".png")
#     plt.close()
#     # plt.scatter(ds[:,0],ds[:,1],c=labels_3)
#     # plt.title("DBSCAN")
#     # plt.savefig("DBSCAN"+str(i)+".png")
#     # plt.close()
#     i += 1
i = 25
ds = np.array(X_new)
vdb = VDBSCAN(kappa= .1)
labels_1 = vdb.fit(ds, eta=0.7).labels_
HDB = HDBSCAN(min_samples=10)
labels_2 = HDB.fit(ds).labels_
labels_3 = DBSCAN(min_samples=5, eps=.5).fit(ds).labels_
plt.scatter(ds[:, 0], ds[:, 1], c=labels_1)
plt.title("vdb")
plt.savefig("vdb"+str(i)+".png")
plt.close()
plt.scatter(ds[:, 0], ds[:, 1], c=labels_2)
plt.title("HDB")
plt.savefig("HDB"+str(i)+".png")
plt.close()
plt.scatter(ds[:,0],ds[:,1],c=labels_3)
plt.title("DBSCAN")
plt.savefig("DBSCAN"+str(i)+".png")
plt.close()
