#Importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

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
#for i in range(n):
#    for j in range(i,n):
#        dist_mat[i][j] = manhattan(i,j)
#        dist_mat[j][i] = dist_mat[i][j]  

dist_pd = pd.read_csv('./sample_data/dist_mat.csv', header=None)
dist_mat = dist_pd.values

eps = 3
minpts = 10

# print(dist_mat)

data['labels'] = -1
data['assignment'] = 0

def dbscan():
    ''' Generates all the clusters along with their corresponding cluster number makes use of Rangeq function '''
    c = 0
    for i in range(n):
        if data['labels'][i] != -1: continue
        Nbs = Rangeq(i)
        if len(Nbs) < minpts:
            data['labels'][i] = 0
            continue
        c = c+1
        data['labels'][i] = c
        seed = Nbs.difference(set([i]))
        
        for j in seed:
            if data['labels'][j]==0: data['labels'][j] = c
            if data['labels'][j]!=-1:continue
            data['labels'][j] = c
            Nbs1 = Rangeq(j)
            if len(Nbs1) >= minpts:
                seed.union(Nbs1)

def Rangeq(i):
    '''Calculate all the neighbours of a given point and returns it as a set'''
    Nbs = set([])
    for j in range(n):
        if dist_mat[i][j]<=eps:
            Nbs.add(j)
    return Nbs

#calling the dbscan function
dbscan()


data['assignment'][data.labels>0] = 1
# data['assignment'].value_counts()

#Plotting the clusters
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

data_copy = data.copy()
data_copy.drop(['labels','assignment'],axis=1,inplace=True)
principalComponents = pca.fit_transform(data_copy)
principalDf = pd.DataFrame(data=principalComponents, columns=['pc1','pc2'])

finalDf = pd.concat([principalDf, data[['assignment']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('pc1', fontsize = 15)
ax.set_ylabel('pc2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r','b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['assignment'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
               , finalDf.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(['Noise', 'Core/Border'])
ax.grid()

#Labelling the noise points as 1 and core/border points as 0
for i in range(n):
    if y[i]==0: 
        y[i] = 1
    else:
        y[i] = 0

pd.concat([y, data[['assignment']]], axis = 1)

#Printing the final accuracy
accuracy = sum(1-abs(data['assignment']-y))/n
#print(accuracy*100)
print(f"Accuracy: {accuracy*100}")
