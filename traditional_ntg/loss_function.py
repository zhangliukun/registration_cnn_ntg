import cv2
import numpy as np
import matplotlib.pyplot as plt

#计算weight的时候isconj为True
def deriv_filt(I,isconj):

    # 使用两者滤波核都可以，但实验表明，使用1*3维的滤波核效果更加好，残差错误更加低
    # if not isconj:
    #     h1 = np.array([[-0.5, 0, 0.5],[-0.5,0,0.5],[-0.5,0,0.5]])
    #     h2 = np.array([[-0.5,-0.5,-0.5],[0,0,0],[0.5,0.5,0.5]])
    # else:
    #     h1 = np.array([[0.5, 0, -0.5],[0.5, 0, -0.5],[0.5, 0, -0.5]])
    #     h2 = np.array([[0.5,0.5,0.5], [0,0,0], [-0.5,-0.5,-0.5]])
    #
    # Ix = cv2.filter2D(I,-1,h1)
    # Iy = cv2.filter2D(I,-1,h2)

    if not isconj:
        h1 = np.array([-0.5, 0, 0.5])
        h2 = np.array([[-0.5],[0],[0.5]])
    else:
        h1 = np.array([0.5, 0, -0.5])
        h2 = np.array([[0.5], [0], [-0.5]])

    Iy = cv2.filter2D(I,-1,h1)
    Ix = cv2.filter2D(I,-1,np.transpose(h2))

    return Ix,Iy


# def inverse_affine_param(p):
#     try:
#         A = np.array([[p[0,0],p[0,1],p[0,2]],[p[1,0],p[1,1],p[1,2]],[0,0,1]])
#         B = np.linalg.inv(A)
#         return B
#     except TypeError:
#         print("type Error")


def affine_transform(im,p):
    height = im.shape[0]
    width = im.shape[1]
    im = cv2.warpAffine(im,p,(width,height))
    return im


def partial_deriv_affine(I1,I2,p,h):

    [H,W] = I1.shape
    x,y = np.meshgrid(range(0,W),range(0,H))
    x2 = p[0,0] * x + p[0,1]*y + p[0,2]
    y2 = p[1,0] * x + p[1,1]*y + p[1,2]
    B = (x2 > W-1) | (x2 < 0) | (y2 > H-1) | (y2 < 0)

    warpI2 = cv2.warpAffine(I2,p,(I2.shape[1],I2.shape[0]))
    I2x,I2y = deriv_filt(I2,False)

    Ipx = cv2.warpAffine(I2x,p,(I2x.shape[1],I2x.shape[0]))
    Ipy = cv2.warpAffine(I2y,p,(I2y.shape[1],I2y.shape[0]))
    It = warpI2 - I1

    It[B] = 0
    Ipx[B] = 0
    Ipy[B] = 0
    return It,Ipx,Ipy

def func_rho(x,order,epsilon=0.01):
    if order == 0:
        y = np.sqrt(x*x + epsilon*epsilon)
        y = np.sum(y)
    elif order == 1:
        y = x/np.sqrt(x*x + epsilon*epsilon)
    else:
        print("Tag | wrong order")
    return y


def ntg(img1,img2):
    [g1x,g1y] = deriv_filt(img1,False)
    [g2x,g2y] = deriv_filt(img2,False)

    m = func_rho(g1x - g2x,0) + func_rho(g1y - g2y,0)
    n = func_rho(g1x,0) + func_rho(g2x,0) + func_rho(g1y,0) + func_rho(g2y,0)
    #y = m/(n+0.01)## TOdo
    y = m/(n+1e-16)## TOdo
    return y



# 返回仿射变换参数p的NTG的梯度
# this.images(:,:,1) = fr1;
# this.images(:,:,2) = fr2;
# this.deriv_filter: drivative kernel in x direction
def ntg_gradient(objdict,p):
    options = objdict['options']
    images = objdict['images']
    #warpI = cv2.warpAffine(images[:,:,1],p,(images[:,:,1].shape[1],images[:,:,1].shape[0]))
    warpI = affine_transform(images[:,:,1],p)

    [It,Ipx,Ipy] = partial_deriv_affine(images[:,:,0],images[:,:,1],p,options['deriv_filter']) # It：warp和I1差值。 Ipx和Ipy是I2的横向纵向梯度

    # print(p)
    # plt.figure()
    # plt.imshow(warpI, cmap=plt.cm.gray_r)
    # plt.figure()
    # plt.imshow(It, cmap=plt.cm.gray_r)
    # plt.figure()
    # plt.imshow(Ipx, cmap=plt.cm.gray_r)
    # plt.figure()
    # plt.imshow(Ipy, cmap=plt.cm.gray_r)
    # plt.show()

    J = ntg(warpI,images[:,:,0])

    [Itx,Ity] = deriv_filt(It,False)
    rho_x = func_rho(Itx,1) - J*func_rho(Ipx,1)
    rho_y = func_rho(Ity,1) - J*func_rho(Ipy,1)

    [wxx,wxy] = deriv_filt(rho_x,True)
    [wyx,wyy] = deriv_filt(rho_y,True)
    w = wxx + wyy

    g = np.zeros((6, 1));
    g[0] = np.mean(w * objdict['X'] * Ipx);
    g[1] = np.mean(w * objdict['Y'] * Ipx);
    g[2] = np.mean(w * Ipx);
    g[3] = np.mean(w * objdict['X'] * Ipy);
    g[4] = np.mean(w * objdict['Y'] * Ipy);
    g[5] = np.mean(w * Ipy);

    g = g.reshape(2,3)

    return g

