#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

k = 7

#Loading data into data(dataframe) and the data sampled has 3000 rows
data = pd.read_csv('./sample_data/data1000.csv')


n = len(data)
#y holds the values of class column
y = data['Class']
data.drop('Class',axis=1,inplace=True)


# Normalization (min max)
for col in data.columns:
    data[col] = ((data[col]-data[col].min())/(data[col].max()-data[col].min()))



#Declaring distance matrix of size n*n
n = len(data)
dist_mat = np.ndarray((n,n))


def manhattan(a_index,b_index):
    '''
    Funtion to calculate the manhattan distance between two rows of the matrix
    '''
    return sum(abs(data.loc[a_index] - data.loc[b_index]))


#Filling the distance matrix with the respective distance values took about 4 minutes

# for i in range(n):
#     for j in range(i,n):
#         dist_mat[i][j] = manhattan(i,j)
#         dist_mat[j][i] = dist_mat[i][j] 
dist_pd = pd.read_csv('./sample_data/dist_mat.csv', header=None)
dist_mat = dist_pd.values

# print(dist_mat)


def k_dist(x):
    '''
    Function to compute the distance of kth nearest neighbour.
    '''
    arr = np.unique(dist_mat[x])
    arr = sorted(arr)
    return arr[k]

def Rangeq(a):
    """Function to calculate all the neighbours of a given point"""
    Nbs = set([])
    for j in range(n):
        if a!=j and dist_mat[i][j]<=k_dist(a):
            Nbs.add(j)
    return Nbs


def reachdist(a,b):
    '''
    Function to compute the reachability distance of
    b from a.
    reach_dist(a,b) = max(distk(b), dist(a,b)) 
    '''
    return max(k_dist(b), dist_mat[a][b])

# values of reachability distance of all pairs are stored in this array
reach_mat = np.ndarray((n,n))

for i in range(n):
    for j in range(n):
        reach_mat[i][j] = reachdist(i, j)


def lrd(a):
    '''
    Function to compute the Linear Reachability Density of point a.
    LRDk(a) = ||N|| / (sum(reachdist(a,b)))  [b belongs to neghbours of a]
    
    '''
    Nbs = Rangeq(a)
    N = len(Nbs)
    summation = 0
    for o in Nbs:
        summation += reach_mat[a][o]
    
    return N/summation


# LRDs of all points are stored in this array
lrdarr = [0]*n

for i in range(n):
    lrdarr[i] = lrd(i)


def LOF(a):
    '''
    This function calculates the Local Outlier Factor of a point a.
    LOF(a) = sum((lrdk(b))/(lrdk(a))) / ||Nk||
     '''
    Nbs = Rangeq(a)
    N = len(Nbs)
    summation = 0
    for o in Nbs:
        summation += lrdarr[o]
    return summation/(N*lrdarr[a])


# This array stores LOF of all points.
lofarr = [-1]*n

data['assignment'] = -1

# calculating lof of all points and deciding if a point is outlier or not.
for i in range(n):
    if lofarr[i]==-1:
        lofarr[i] = LOF(i)
        
    if lofarr[i]>1.9:
        data['assignment'][i] = 1
    else:
        data['assignment'][i] = 0
     

# Accuracy is calculated using labels given in the dataset's last column
accuracy = sum(1-abs(data['assignment']-y))/n
print(f"Accuracy: {accuracy*100}")


# attributes are reduced to get a 2D plot.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

data_copy = data.copy()
data_copy.drop(['assignment'],axis=1,inplace=True)
principalComponents = pca.fit_transform(data_copy)
principalDf = pd.DataFrame(data=principalComponents, columns=['pc1','pc2'])

finalDf = pd.concat([principalDf, data[['assignment']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('pc1', fontsize = 15)
ax.set_ylabel('pc2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['b','r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['assignment'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
               , finalDf.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(['Core/Border','Noise'])
ax.grid()