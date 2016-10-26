import numpy as np
import os, sys, pickle, re, time, copy
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
os.environ['GLOG_minloglevel'] = '2'
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
caffe_root = '/home/mo/caffe/'
sys.path.append(caffe_root+'python')
import caffe
caffe.set_mode_gpu()

""" load net weights and allocate memory """
model_def = '/home/mo/caffe/models/places/places205CNN_deploy.prototxt'
W = '/home/mo/caffe/models/places/places205CNN_iter_300000.caffemodel'
net = caffe.Net(model_def, W, caffe.TEST)
net.blobs['data'].reshape(10, 3, 227, 227)

""" pre-processing for input """
blob = caffe.proto.caffe_pb2.BlobProto()
mu_f = open(caffe_root+'models/places/places205CNN_mean.binaryproto','rb')
blob.ParseFromString(mu_f.read())
mu = np.array(caffe.io.blobproto_to_array(blob))[0]
BGRmean = mu.mean(1).mean(1)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', BGRmean)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

image = caffe.io.load_image('../cam/still_3_16/IMG_20160316_154023.jpg')
net.blobs['data'].data[0] = transformer.preprocess('data',image)
output = net.forward()

filters = net.params['fc6'][0].data
t = filters.reshape(4096,256,6,6)

cmap = plt.cm.Reds
x, y, z = np.meshgrid(range(256),range(6),range(6))
x = x.flatten().tolist()
y = y.flatten().tolist()
z = z.flatten().tolist()

ims = []
fig = plt.figure()
ax = Axes3D(fig)
for i in range(0,t.shape[0],100):
	# ax.cla()
	ax.axis('off')
	# ax.azim -= 5
	cs = cmap(t[i].flatten() / t[i].max())
	h1 = ax.scatter(y,z,x,c=cs.tolist(),alpha=.1,marker='s',s=100,lw=0)
	h2 = ax.text(.5,.5,.95, '... neuron #%d'%(i+1))
	ims.append([h1,h2])
	# plt.pause(.1)

ani = anim.ArtistAnimation(fig, ims)
print "saving animation to file"
ani.save('out.mp4', writer="avconv", fps=2)
