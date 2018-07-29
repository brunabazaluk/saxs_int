def read_const():

	cake = open('cake_pars.py')
	cakestr = cake.read().split('\n')
	len_cake = len(cakestr)


	geo = open('geometry_pars.py')
	geostr = geo.read().split('\n')
	len_geo = len(geostr)

	const = []

	for i in range(len_cake):
		index = cakestr[i].find(' ')
		c = cakestr[i][:index]
		c=c.split('=')
		if len(c) == 2:
			const.append(c)


	for i in range(len_geo):
		index = geostr[i].find(' ')
		c = geostr[i][:index]
		c=c.split('=')
		if len(c) == 2:
			const.append(c)

	return const