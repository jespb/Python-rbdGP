import pandas

from rbdgp.RbdGP import RbdGP

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-rbdGP
#
# Copyright ©2023-2023 J. E. Batista
#



filename= "heart.csv"

# Open the dataset
ds = pandas.read_csv("datasets/"+filename)
class_header = ds.columns[-1]

# Split the dataset
Tr_X, Te_X, Tr_Y, Te_Y = train_test_split(ds.drop(columns=[class_header]), ds[class_header], 
		train_size=0.7, random_state = 42, stratify = ds[class_header])

# Train a model
model = RbdGP()
model.fit(Tr_X, Tr_Y)

# Predict test results
pred = model.predict(Te_X)

# Obtain test accuracy
print( accuracy_score(pred, Te_Y) )

