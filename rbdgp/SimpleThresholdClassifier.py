from statistics import median



# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-rbdGP
#
# Copyright ©2023-2023 J. E. Batista
#


 
class SimpleThresholdClassifier:

	threshold = None
	sign = 0
	classes = []

	def __init__(self, threshold = 0):
		self.threshold = threshold

	def fit(self,X=None,Y=None):
		Y = list(Y)
		self.classes = list(set(Y))
		cl1 = []
		cl2 = []
		X = list(X.iloc[:,0])
		for xi in range(len(X)):
			if Y[xi] == self.classes[0]:
				cl1.append(X[xi])
			else:
				cl2.append(X[xi])
		m1 = median(cl1)
		m2 = median(cl2)
		self.threshold = (m1+m2) / 2
		if m2 > m1:
			self.sign=1


	def predict(self, X):	
		"""
		Receives X, a 1-D array of real values
		Return a list of predictions based on the value
		"""	
		predictions = []
		for v in list(X.iloc[:,0]):
			if self.sign == 0:
				predictions.append( self.classes[0] if v > self.threshold else self.classes[1])
			else:
				predictions.append( self.classes[0] if v < self.threshold else self.classes[1])
		return predictions

