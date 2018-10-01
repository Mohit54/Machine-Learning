import pandas as pd
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
# load all data
data1 = read_csv('datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data2 = read_csv('datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data3 = read_csv('datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)

#Stack Vertically and maintain temporal order
data = pd.concat([data1, data2, data3])
#drop ROw No
data.drop('no', axis=1, inplace=True)
#Save Combined Dataset
data.to_csv('combined.csv')

#load the dataset
data = read_csv('combined.csv', header=0,index_col=0,parse_dates=True,squeeze=True)
values=data.values

#Split data into inputs & Outputs
X,y = values[:,:-1], values[:,-1]
#SPlit the dataset
trainX, testX, trainy, testy=train_test_split(X,y, test_size=0.3, shuffle=False, random_state=1)

# determine the number of features
'''n_features = data1.values.shape[1]
pyplot.figure()
for i in range(1, n_features):
	# specify the subpout
	pyplot.subplot(n_features, 1, i)
	# plot data from each set
	pyplot.plot(data1.index, data1.values[:, i])
	pyplot.plot(data2.index, data2.values[:, i])
	pyplot.plot(data3.index, data3.values[:, i])
	# add a readable name to the plot
	pyplot.title(data1.columns[i], y=0.5, loc='right')
pyplot.show()'''

'''def naive_prediction(testX,value):
	return[value for x in range(len(testX))]

for value in[0,1]:
	#forecast
	yhat = naive_prediction(testX,value)
	#evaluate
	score = accuracy_score(testy,yhat)
	#summarize
	print('Naive=%d score=%.3f'%(value,score))'''

model = LogisticRegression()

#Fit model on training set
model.fit(trainX,trainy)
#predict the test set
yhat = model.predict(testX)
#evaluate model skill
score = accuracy_score(testy,yhat)
print(score)


