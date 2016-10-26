import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import PIL.Image
import os

def deg2dec(d):
	return 1.*d[0][0]/d[0][1]+1.*d[1][0]/d[1][1]/60+1.*d[2][0]/d[2][1]/3600

prepath = '../cam/still_3_16/'
# images = [f for f in os.listdir(prepath) if f.endswith('.jpg')]
# images.sort()

background = mpimg.imread('../localize/map.jpg')
bb = [45.591834,45.595397,-122.462629,-122.456612]
nr, nc, n3 = background.shape

fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(5, 9, forward=True)

with open('points','r') as fr:
	for line in fr:
		images = line.split()
		for image in images:
			ax2.cla()
			ax2.imshow(background)
			ax2.set_xlim([0, nc-1])
			ax2.set_ylim([nr-1, 0])
			img = PIL.Image.open(prepath+image+'.jpg')
			ax1.imshow(np.array(img))
			lat = deg2dec(img._getexif()[34853][2])
			lon = deg2dec(img._getexif()[34853][4]) * -1
			x = 0  + (lon-bb[2])/(bb[3]-bb[2])*nc
			y = nr - (lat-bb[0])/(bb[1]-bb[0])*nr
			ax2.plot(x,y,'ro')
			ax2.set_title('image={0}'.format(image))
			plt.pause(1)
