import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math 


def main():
	#getting variables (these are set as examples, probably they'll be taken from files)
	initial_angle = 10
	final_angle = 350
	delta_q = 1

	#getting data from .tif
	img=str(input('filename: '))

	data = Image.open(img)
	data = np.array(data)

	#coord. invertidas ou n?
	data = pd.DataFrame(data)

	row=len(data.index)
	col = len(data.columns)

	polar_coor, qs = polar(data, row, col)

	Idataframe = select_I(qs, delta_q)

	Idataframe = uncertainty(Idataframe)

	print_graph(Idataframe)

#converting to polar coordinates
#here i'll have the list of all I's per slice (function of radius and angle)
def polar(data, x_size, y_size):
	#https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan.html
	#about arctan numpy function

	polar_data = pd.DataFrame(index=range(x_size),columns=range(y_size))
	radius_dict = {}

	theta=0
	r=0

	for x in range(1, x_size):
		for y in range(1, y_size):
			r = np.sqrt(np.power(x, 2) + np.power(y, 2))
			theta = np.arctan(x/y)
			I = data.iat[x,y]
			polar_data.iat[x, y] = [r, theta, I]
			radius_dict.update({r : I})

	#polar_data is a dataframe and each cell[i,j] is a list [r, theta, I], r and theta are i and j's funcions 
	return polar_data, radius_dict


#building I vector with all I's per "slice" (delta_q)
def select_I(qdict, delta_q):

	qlist = sorted(qdict.items(), key = lambda x : x[0])
	# qlist is now an ordered list of tuples (key, value)
	
	l = len(qlist)
	max_q = qlist[l - 1][0]
	q = np.float64(qlist[0][0])
	I_df = pd.DataFrame(columns=['start', 'end', 'I_s'])
	Ilist = []

	for i in range(l):

		if (qlist[i][0] >= q) and (qlist[i][0] < q+delta_q): 
			Ilist.append(qlist[i][1])

		else:
			row = pd.DataFrame([[q, q+delta_q, Ilist]])
			I_df.append(row, ignore_index=True)
			q += delta_q
			Ilist = []

	return I_df

#creating uncertainty vector using numpy functions to calculate mean and std dev
def uncertainty(I):
	#I is the dataframe with all I's per "slice" (delta_q)
	#I = [[q0, q, [I0, I1, ...]], [q, q1, [I0, I1, ...]]...]

	I_avg = []
	I_std = []
	I_meanstd = []

	l = len(I)

	for i in range(l):

		i0 = I[i][2]
		stand_dev = np.std(i0, dtype=np.float64) #float64 to make it more accurate
		I_meanstd.append(stand_dev/(sqrt(len(i0))))
		I_avg.append(np.mean(i0))
		I_std.append(stand_dev)
	

	s1 = pd.Series(I_avg)
	s2 = pd.Series(I_std)
	s3 = pd.Series(I_meanstd)

	I['Average I'] = s1.values
	I['Is standart deviation'] = s2.values
	I['Average I standart deviation'] = s3.values

	return I

def print_graph(df):
	#print qxI graph showing std_dev

	df.plot(x='Average I', y='q', xerr='Average I standart deviation')

	plt.show()




main()