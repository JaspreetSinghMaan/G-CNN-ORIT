import tensorflow as tf
import numpy as np
import math

""" Computes the PHTs (PCET)
    INPUT: - data_tensor in Z2, a tensorflow Tensor with expected shape:
                [BatchSize, Height, Width, ChannelsIN]
    OUTPUT: - feat, the tensor after computing PHTs with shape
                [BatchSize, Nmax]
""" 
def PCET(data_tensor):
    feat = tf.concat([PCETs(data_tensor,0,0), PCETs(data_tensor,1,0), PCETs(data_tensor,1,1), 
                  PCETs(data_tensor,2,0), PCETs(data_tensor,2,1), PCETs(data_tensor,2,2), 
                  PCETs(data_tensor,3,0), PCETs(data_tensor,3,1), PCETs(data_tensor,3,2), PCETs(data_tensor,3,3),
                  PCETs(data_tensor,4,0), PCETs(data_tensor,4,1), PCETs(data_tensor,4,2), PCETs(data_tensor,4,3), 
                  PCETs(data_tensor,4,4), PCETs(data_tensor,5,0), PCETs(data_tensor,5,1), PCETs(data_tensor,5,2), 
                  PCETs(data_tensor,5,3), PCETs(data_tensor,5,4), PCETs(data_tensor,5,5), 
                  ], 1)
    return feat

def PCETs(p,n,m):
  p = tf.transpose(p,(0,3,1,2))
  N = int(p.shape[2])
  x = np.arange(0,N,1)
  y = np.arange(0,N,1)
  D = N*np.sqrt(2)
  [X,Y] = np.meshgrid(x,y)
  R = np.sqrt((2.*X-N+1)**2+(2.*Y-N+1)**2)/D

  Theta = np.arctan2((2.*Y-N+1)/D, (2.*X-N+1)/D)
  Theta = Theta.transpose(1,0) 
  Theta = ((Theta<0.0)*(2.0*np.pi+Theta))+((Theta>=0.0)*(Theta))

  #Rad = radialpoly(R,n)    # get the radial polynomial
  Rad = np.exp(+1j*2.0*np.pi*n*R*R)
  norm1 = (4.0*(1.0/np.pi))/(D*D)
  ele2 = tf.cast(Rad, tf.complex64)*tf.cast(tf.exp(-1j*m*Theta), tf.complex64)

  Product = tf.cast(p,tf.complex64)*ele2
  Z = (tf.reduce_sum(tf.reduce_sum(Product,axis=3,keepdims=False),axis=2,keepdims=False))
  Z = norm1*Z
  A = tf.abs(Z)
  return A