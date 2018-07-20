import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math 


#getting variables
initial_angle = 10
final_angle = 350

#getting data from .tif
img=str(input('filename: '))

data = Image.open(img)
data = np.array(data)

#coord. invertidas ou n?
data = pd.DataFrame(data)

#converting to polar coordinates
	#here i'll have the list of all I's per slice (function of radius and angle)
def polar(data, x_size, y_size):
	#https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan.html
	#about arctan numpy function

	polar_data = pd.DataFrame(index=range(x_size),columns=range(y_size))
	radius_dict = {}

	theta=0
	r=0

	for x in range(x_size):
		for y in range(y_size):
			r = np.sqrt(np.power(x, 2) + np.power(y, 2))
			theta = np.arctan(x/y)
			I = data.iat[x,y]
			polar_data.iat[x, y] = [r, theta, I]
			radius_dict.update({r : I})

	#polar_data is a dataframe and each cell[i,j] is a list [r, theta, I], r and theta are i and j's funcions 
	return polar_data, radius_dict


#creating uncertainty vector

def uncertainty(I):
	#I is the vector with all I's per "slice"
	#I = [[i00, i01, i02], [i10, i11, i12]...]

	unc = []

	l = len(I)

	for i in range(l):

		s = np.std(I[i])
		se = s/(sqrt(len(I[i])))
		I_avg = np.mean(I[i])

		unc.append([I_avg, s, se])

	return unc

