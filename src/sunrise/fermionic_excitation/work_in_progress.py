import tequila as tq
import sunrise as sun
import numpy as np

a = np.array([[0,1],[2,3]])
print(a.reshape(()))
idx = [1,3]
n = np.array([[4,5],[6,7]])
jdx = [0,2]
new_shape = 4
n = np.zeros(shape=(new_shape,new_shape))