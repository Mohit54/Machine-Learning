from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

xs = [1,2,3,4,5]
ys = [5,4,6,5,6]

xs = np.array([1,2,3,4,5], dtype = np.float64)
ys = np.array([5,4,6,5,6], dtype = np.float64)

def best_fit_slope(xs, ys):
	m = (mean(xs) * mean(ys) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs*xs))
	return m

m = best_fit_slope(xs,ys)
print (m)

def best_fit_slope_and_intercept(xs, ys):
	m = (mean(xs) * mean(ys) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs*xs))

	b = mean(ys) - m*mean(xs)
	return m,b


m,b = best_fit_slope_and_intercept(xs, ys)
print(m,b)

#regression_line = [(m*x)+b ]
regression_line = []
for x in xs:
    regression_line.append((m*x)+b)

"""plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, regression_line)
plt.show()"""


predict_x = 7

predict_y = (m*predict_x + b)
print(predict_y)

plt.scatter(xs,ys,color='#003F72', label='data')
plt.plot(xs, regression_line, label='regression line')
plt.scatter(predict_x,predict_y, color='#000000', label = 'prediction')
plt.legend(loc=4)
plt.show()