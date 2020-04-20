from scipy import interpolate
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ntg_pytorch.register_func import scale_image

def readImages(u1,v1):
	try:
		u = np.asarray(Image.open(u1).convert('L'), dtype=np.float64)
		u=u/np.max(u)
		v = np.asarray(Image.open(v1).convert('L'), dtype=np.float64)
		v=v/np.max(v)
		return (u,v)
	except IOError:
		return -1

def interpolation(v):
	k=np.shape(v)[0]
	k1=np.shape(v)[1]
	return interpolate.interp2d(np.arange(k),np.arange(k1),v.T,kind='cubic',fill_value=0)


if __name__ == '__main__':
	# img1 = io.imread('../datasets/row_data/multispectral/mul_1s_s.png')
	# img2 = io.imread('../datasets/row_data/multispectral/mul_1t_s.png')

	img1 = '../datasets/row_data/multispectral/mul_1s_s.png'
	img2 = '../datasets/row_data/multispectral/mul_1t_s.png'

	u,v = readImages(img1,img2)

	plt.figure()
	plt.imshow(v,cmap='gray')

	image_data = interpolation(v)

	image_data_z = np.reshape(image_data.z,u.shape)

	plt.figure()
	plt.imshow(image_data_z,cmap='gray')
	plt.show()