import pickle, os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10) 
plt.rcParams['image.interpolation'] = 'nearest'

def visualize(data,name):
  for m in range(data.shape[0]):
    if data[m].max()!=data[m].min():
      data[m] = (data[m]-data[m].min())/(data[m].max()-data[m].min())
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0,n*n-data.shape[0]), (0,1), (0,1))+((0,0),)*(data.ndim-3)
  data = np.pad(data, padding, mode='constant', constant_values=1)
  data = data.reshape((n, n) + data.shape[1:])
  data = data.transpose((0,2,1,3)+(4,)*(data.ndim-4))
  data = data.reshape((n*data.shape[1],n*data.shape[3])+data.shape[4:])
  plt.imshow(data)
  plt.axis('off')
  plt.pause(1)
  plt.savefig('../conv3-landmarks/'+name, bbox_inches='tight')

layer = 'conv3'

with open('../localize/net_testing_'+layer) as fr:
	X = pickle.load(fr)

for j in range(X.shape[0]):
	visualize(X[j],'frame_%03d.jpg'%(j+1))
