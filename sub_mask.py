import fabio
import numpy as np

img=str(input('filename: '))

data = fabio.open(img)
data = np.array(data.data)



print(data.dim1)