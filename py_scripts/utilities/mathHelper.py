import numpy as np
import scipy.signal as signal
from math import cos, sin, pi, ceil
		
def radians (array):
	return array*pi/180

# turn euler angle into vector
def euler2Vec (vec3):
	rotx = vec3[0]
	roty = vec3[1]
	rotz = vec3[2]
	x = np.sin(radians(roty))*np.cos(radians(rotx))
	y = -np.sin(radians(rotx))
	z = np.cos(radians(roty))*np.cos(radians(rotx))
	return np.array([x,y,z])

# calculate angle between two vector
def calcAngleDeg (v1,v2):
	# v1*v2/(sqrt(v1*v1)*sqrt(v2*v2))
	costheta = np.sum(np.multiply(v1,v2),axis=1)/(
		np.sqrt(np.sum(np.multiply(v1,v1),axis=1))*
		np.sqrt(np.sum(np.multiply(v2,v2),axis=1)))
#     if np.any(abs(costheta)>1):
#         print("costheta value wrong:\n")
#         print(costheta[(abs(costheta)>1)])
	return 180/pi*np.arccos([min(max(x,-1),1) for x in costheta])

def smooth(y, box_pts = 7, smooth_type = 'box',sigma = 3):
	if smooth_type == 'box':
		box = np.ones(box_pts)/box_pts
	elif smooth_type == 'gaussian':
		box = signal.gaussian(box_pts,sigma)
		box = box/np.sum(box)
#         print(box)
	y_smooth = np.convolve(y, box, mode='valid')
	return y_smooth