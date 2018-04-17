from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

iris = datasets.load_iris().data.ravel()
iris = np.split(iris,len(iris)/4)
C = np.random.random((3,4))*8
error = 1
itter = 0

while error != 0:
	a = np.empty([len(iris), 3])
	old  = np.copy(C)
	
	# Calculate distance from each point to each center
	for j in range(3):
		a[:,j] = (np.linalg.norm(iris-C[j], axis = 1))**2

	# Determine which center each point is closest to
	b = a.argmin(axis=1)

	# Move centers
	for j in range(3):
		c = np.array([]).reshape(0,4)

		# Create array of points associated with each center
		for k in range(len(iris)):
			if b[k] == j:
				c = np.vstack([c,iris[k]])

		# Move center
		if len(c) == 0:
			C[j] = np.random.random((1,4))*8
		else:
			C[j] = c.mean(axis=0)
			
	error = (old-C).sum()
	itter += 1

print(itter)
print(C)