import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math 


#getting variables
initial_angle = 
final_angle = 

#getting data from .tif
img=str(input('filename: '))

data = Image.open(img)
data = np.array(data)

#coord. invertidas ou n?
data = pd.DataFrame(data)

#converting to polar coordinates
	#here i'll have the list of all I's per slice (function of radius and angle)





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

