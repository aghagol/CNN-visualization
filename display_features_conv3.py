import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['figure.figsize'] = (10, 10) 
plt.rcParams['image.interpolation'] = 'nearest'
from matplotlib.patches import Rectangle

def visualize(data):
  for m in range(data.shape[0]):
    if data[m].max()!=data[m].min():
      data[m] = (data[m]-data[m].min())/(data[m].max()-data[m].min())
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0,n*n-data.shape[0]), (0,1), (0,1))+((0,0),)*(data.ndim-3)
  data = np.pad(data, padding, mode='constant', constant_values=1)
  data = data.reshape((n, n) + data.shape[1:])
  data = data.transpose((0,2,1,3)+(4,)*(data.ndim-4))
  return data.reshape((n*data.shape[1],n*data.shape[3])+data.shape[4:])

def coor(bb): #extract coordinates for bbox
  cors = ( bb[0][0] , bb[1][0] )
  rows = bb[0][1] - bb[0][0]
  cols = bb[1][1] - bb[1][0]
  return (cors,rows,cols)

layer = 'conv3'

# images = [f for f in os.listdir('../cam/still_3_16/') if f.endswith('.jpg')]
# images.sort()
images = []
with open('points','r') as fpoints:
  for line in fpoints:
    images += line.split()[0:3]
print 'list of images:'
print '\n'.join(images)

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(1,1,1)
ax1 = fig.add_axes([.85,.0,.1,.1])
ax0.set_title('layer: conv3')

for image in images:
  nail = mpimg.imread('../cam/still_3_16/'+image+'.jpg')
  with open('features/'+image+'.meta','rb') as fmeta:
    meta = pickle.load(fmeta)
  with open('features/'+image+'.'+layer,'rb') as fr:
    Y = pickle.load(fr)
  for j in range(len(meta)):
    ax0.cla()
    ax0.imshow(visualize(Y[j]))
    ax0.axis('off')
    ax1.cla()
    ax1.imshow(nail)
    ax1.axis('off')
    bbcors, bbrows, bbcols = coor(meta[j])
    ax1.add_patch( Rectangle(bbcors[::-1], bbcols, bbrows,
      fill=False, linewidth=1.5, edgecolor="red") )
    fig.tight_layout()
    fig.savefig('../conv3/%s_p%02d.png'%(image,j))
    plt.pause(1)
