import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math 
import fabio

import cake_pars
import geometry_pars

def main():

	#getting data from .tif
	img=str(input('filename: '))
	mask = str(input('mask file: '))

	data = Image.open(img)
	data = np.array(data)

	mask = fabio.open(mask)
	mask = np.array(mask.data)

	#coord. invertidas ou n?
	data = pd.DataFrame(data)
	delta_q = (cake_pars.outrad - cake_pars.inrad)/cake_pars.nrad

	row=len(data.index)
	col = len(data.columns)

	polar_coor, qs = polar_q(data, row, col)


	Idataframe = select_I(mask, qs, polar_coor, delta_q)


	Idataframe = uncertainty(Idataframe)

	#create_file(Idataframe)

	print_graph(Idataframe)

#converting to polar coordinates
#here i'll have the list of all I's per slice (function of radius and angle)
def polar_q(data, x_size, y_size):
	#https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan.html
	#about arctan numpy function

	polar_data = pd.DataFrame(index=range(x_size),columns=range(y_size))
	q_list = []

	theta=0
	d=0

	#usar 0 do feixe direto!!!!
	for x in range(1, x_size):
		for y in range(1, y_size):
			d = np.sqrt(np.power((x - geometry_pars.x0), 2) + np.power((y - geometry_pars.y0), 2)) #uses (x0,y0) as origin
			if d >= cake_pars.inrad and d <= cake_pars.outrad:	

				theta = np.arctan((y - geometry_pars.y0)/(x - geometry_pars.x0)) #in rads
				theta = (theta*180)/np.pi #converts to grades
				
				if theta >= cake_pars.s_az and theta <= cake_pars.e_az:

					I = data.iat[x,y]
					r = d*(geometry_pars.px/1000) #converts from pixel to milimeter!!! 

					alpha = (np.arctan(r/geometry_pars.dis))/2 #alpha is the angle used to get q
					q = (4*np.pi*np.sin(alpha))/geometry_pars.wvl

					polar_data.iat[x, y] = [x, y, q, theta, I]
					q_list.append((q, I, x, y))



	#polar_data is a dataframe and each cell[i,j] is a list [r, theta, I], r and theta are i and j's funcions 
	return polar_data, q_list


#building I vector with all I's per "slice" (delta_q)
def select_I(mask, qdict, polar, delta_q):

	qlist = sorted(qdict, key = lambda x : x[0])
	# qlist is now an ordered list of tuples (key, value)

	delta_q = np.float64(0.001)
	l = len(qlist)
	max_q = qlist[l - 1][0]
	q = np.float64(qlist[0][0])	
	I_df = pd.DataFrame(columns=['start', 'end', 'I_s'])
	Ilist = []
	
	for i in range(1, l):
		
		if (np.float64(qlist[i][0]) >= np.float64(q)) and (np.float64(qlist[i][0]) < np.float64(q+delta_q)): 
			if mask[qlist[i][2]][qlist[i][3]] == 1:
				Ilist.append(qlist[i][1])

		else:
			row = pd.DataFrame([[q, q+delta_q, Ilist]], columns=['start', 'end', 'I_s'])
			I_df = pd.concat([I_df, row], ignore_index=True)
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

		i0 = np.array(I.iat[i, 2])
		

		if len(i0) > 1: 
			stand_dev = np.std(i0)
			I_meanstd.append(stand_dev/(np.sqrt(len(i0))))
			I_avg.append(np.mean(i0))
			I_std.append(stand_dev)
		elif i0 != []:
			I_meanstd.append(0)
			I_avg.append(np.float64(i0))
			I_std.append(0)
				
	

	s1 = pd.Series(I_avg)
	s2 = pd.Series(I_std)
	s3 = pd.Series(I_meanstd)

	newI = pd.DataFrame()
	newI['Average I'] = s1.values
	newI['Is standart deviation'] = s2.values
	newI['Average I standart deviation'] = s3.values
	
	I = pd.concat([I, newI], axis=1)

	return I

def print_graph(df):
	#print qxI graph showing std_dev
	print(df)
	#df.astype(float)
	df.plot(x='start', y='Average I', logy=True, yerr='Average I standart deviation', ecolor='r')
	#p.set_yscale('log')
	plt.show()


def create_file(df):
	file = open('bla.dat', 'w+')
	file.write('q            I(q)            Is_stddev\n')
	for i in range(len(df)):
		file.write(str(df.iloc[i]['start'])+'            '+str(df.iloc[i]['Average I'])+'            '+str(df.iloc[i]['Average I standart deviation'])+'\n')
	file.close()

main()