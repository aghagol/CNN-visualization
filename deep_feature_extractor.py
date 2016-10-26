import numpy as np
import os, sys, pickle, re, time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
os.environ['GLOG_minloglevel'] = '2'
caffe_root = '/home/mo/caffe/'
sys.path.append('/home/mo/caffe/python')
import caffe
caffe.set_mode_gpu()

campath = '../cam/still_3_16/'
featpath = 'features/'
if not os.path.exists(featpath): os.mkdir(featpath)
def bboxes(imsize):
	bbs = [] 
	for scale in [.5, .65, .8, .9]:
		rows_c 	= (int((1-scale)*imsize[0]/2),int((1+scale)*imsize[0]/2))
		cols_l 	= (0,int(scale*imsize[1]))
		cols_c 	= (int((1-scale)*imsize[1]/2),int((1+scale)*imsize[1]/2))
		cols_r 	= (int((1-scale)*imsize[1]),imsize[1])
		bbs.append( ( rows_c , cols_l ) )
		bbs.append( ( rows_c , cols_c ) )
		bbs.append( ( rows_c , cols_r ) )
	bbs.append( ( (0,imsize[0]) , (0,imsize[1]) ) )
	return bbs

def display_bbs(bbs):
	fig, ax = plt.subplots(1,1)
	for t in bbs:
		print t
		b1 = (t[1][0], t[0][0])
		b2 = t[1][1]-t[1][0]
		b3 = t[0][1]-t[0][0]
		ax.add_patch(Rectangle(b1,b2,b3,fill=False,edgecolor="red"))
		ax.set_xlim([-10, 2010])
		ax.set_ylim([-10, 2010])
		plt.pause(5)

# layers = ['data','pool1','pool2','conv3','conv4','conv5','fc6','fc7']
layers = ['conv3','fc6']
K = len(bboxes((100,100))) #number of patches per image
batch_size = 50

images = [f for f in os.listdir(campath) if f.endswith('.jpg')]
images.sort()
N = len(images)

""" load net weights and allocate memory """
model_def = '/home/mo/caffe/models/places/places205CNN_deploy.prototxt'
W = '/home/mo/caffe/models/places/places205CNN_iter_300000.caffemodel'
net = caffe.Net(model_def, W, caffe.TEST)
net.blobs['data'].reshape(batch_size*K, 3, 227, 227)

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

""" forward step """
for i in range(N / batch_size + 1):
	for j in range(batch_size):
		n = batch_size*i + j
		if (n>=N): break
		print 'CPU: working on %s (%.2f%%)'%(images[n],(n+1.)/N*100)
		image = caffe.io.load_image(campath+images[n])
		bbs = bboxes(image.shape)
		with open(featpath+images[n][:-4]+'.meta','wb') as fbb:
			pickle.dump(bbs,fbb)
		for k, bb in enumerate(bbs):
			print '...patch number %d'%(k+j*K)
			net.blobs['data'].data[k+j*K] = transformer.preprocess('data',
				image[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1],:])
	print 'Titan enters'
	t1 = time.time()
	output = net.forward()
	t2 = time.time()
	print 'Titan leaves (%f ms / patch)' %((t2-t1)*1000/batch_size/K)
	print 'CPU: writing to disk'
	for j in range(batch_size):
		n = batch_size*i + j
		if (n>=N): break
		for layer in layers:
			with open(featpath+images[n][:-4]+'.'+layer,'wb') as fw:
				pickle.dump(net.blobs[layer].data[range(j*K,j*K+K)],fw)
