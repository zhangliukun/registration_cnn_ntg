# 参考链接https://matthew-brett.github.io/teaching/mutual_information.html

# compatibility with python2
from __future__ import print_function # print方法
from __future__ import division # 1/2==0.5而不是0

# import common modules
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# set gray colormap and nearnst neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

# 注意，如果在IPython中运行的话，使用%matplotlib来使interactive plots生效，如果在Jupyter Notebook中
# 是运行，使用%matplotlib inline

image1 = io.imread("/Users/zale/project/myself/registration_cnn_ntg/datasets/row_data/multispectral/It.jpg")
image2 = io.imread("/Users/zale/project/myself/registration_cnn_ntg/datasets/row_data/multispectral/It180.jpg")

# t1_slice = np.array([11,21,31,41,54,61,71,81,91,101,112,123,131,145,159])
# t2_slice = np.array([15,25,34,45,56,67,77,86,95,108,111,125,136,145,155])

t1_slice = image1[:,:,0]
t2_slice = image2[:,:,1]
#t2_slice = image1[:,:,1]


plt.imshow(np.hstack((t1_slice,t2_slice)))

# plt.figure()
fig,axes = plt.subplots(1,2)
axes[0].hist(t1_slice.ravel(),bins=20)
axes[0].set_title('t1 hist')

axes[1].hist(t2_slice.ravel(),bins=20)
axes[1].set_title('t2 hist')
# plt.show()

plt.figure()
# plt.plot(t1_slice.ravel(),t2_slice.ravel(),'.')
# plt.xlabel('t1 signal')
# plt.ylabel('t2 signal')
# plt.title('t1 vs t2 signal')
hist_2d,x_edges,y_edges = np.histogram2d(t1_slice.ravel(),t2_slice.ravel(),bins=20)
plt.imshow(hist_2d.T,origin='lower')
plt.xlabel('T1 signal bin')
plt.ylabel('T2 signal bin')


# Show log histogram, avoiding divide by 0
plt.figure()
hist_2d_log = np.zeros(hist_2d.shape)
non_zeros = hist_2d != 0
hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
plt.imshow(hist_2d_log.T, origin='lower')
plt.xlabel('T1 signal nozero bin')
plt.ylabel('T2 signal nozero bin')

plt.show()

def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

print(mutual_information(hist_2d))