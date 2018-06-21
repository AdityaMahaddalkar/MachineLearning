import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class LogisticRegression:

	def __init__(self):
		pass

	def sigmoid(self, z):
		return 1 / (1 + np.e**-(z))

	def gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
		'''
		Function used to calculate optimum weights and bias
		'''
		#Step 1 : Initialize variables
		try:
			n_samples, n_features = X.shape
		except:
			n_samples = X.shape[0]
			n_features = 1
			X.shape = (X.shape[0], 1)
		self.weights = np.zeros(shape=(n_features, 1))
		self.bias = 0
		costs = []
		
		for i in range(iterations):

			#Step 2: Compute hypothesis of x by using sigmoid function
			y_predict = self.sigmoid(np.dot(X,self.weights) + self.bias)
			#print(y_predict)
			#Step 3: Compute the cost of current weights and bias
			cost =  (-1/n_samples) * np.sum((y * np.log(y_predict) - 1-y * np.log(1-y_predict)))
			costs.append(costs)

			#Step 4: Compute partial derivatives
			dw = (1/n_samples) * np.dot(X.T, (y_predict - y))
			db = (1/n_samples) * np.sum(y_predict-y)

			#Step 5: Adjust the weights and bias accordingly
			self.weights = self.weights - learning_rate * dw
			self.bias = self.bias - learning_rate * db

			if i % 1000 == 0:
				print("Cost = {} at i = {}".format(cost, i))
                

		return self.weights, self.bias,costs

	def predict_value(self, X):

		return np.dot(X, self.weights) + self.bias


def main():

	model1 = LogisticRegression()

	X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
	y = np.array([[0] , [0], [0], [1]])

	X_train, x_test, Y_train, y_test = train_test_split(X, y)

	w, b, costs = model1.gradient_descent(X, y, learning_rate=0.000001, iterations=10000)
	print(w, b)
	plt.plot(list(range(len(costs))), costs)
	plt.title("Change in cost through iteration")
	plt.xlabel("Iteration")
	plt.ylabel("Cost")
	plt.show()

	
if __name__ == '__main__':
        main()

