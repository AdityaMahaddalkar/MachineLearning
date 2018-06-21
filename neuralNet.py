##Simple neural network code 
'''
	
Input layer: 4 inputs
Hidden layer: 5 terms
Output layer: 1 output

Binary classification problem
'''
import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:

	def __init__(self, x, y):
		self.inputs = x
		self.weights1 = np.random.rand(x.shape[1],5)
		self.weights2 = np.random.rand(5, 1)
		self.output = y
		print("Shape of inputs = {}".format(self.inputs.shape))
		print("Shape of weights1 = {}".format(self.weights1.shape))
		print("Shape of weights2 = {}".format(self.weights2.shape))
		print("Shape of output = {}".format(self.output.shape))


	def sigmoid(self, z):
		return 1 / (1 + np.e**(-z))

	def sigmoid_derivative(self, z):
		return z * (1 - z)

	def forwardprop(self):
		self.layer1 = self.sigmoid(np.dot(self.inputs, self.weights1))
		self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights2))

	def backprop(self):
		self.del_3 = (self.layer2 - self.output)
		self.del_2 = np.dot(self.del_3, self.weights2.T) * self.sigmoid_derivative(self.layer1)

	def costFunction(self):
		return 0.5 * np.sum((self.layer2 - self.output)**2)

	def train(self, learning_rate=0.001, iterations=1000):

		costs = []

		for i in range(iterations):
			self.forwardprop()
			self.backprop()
			dw2 = np.dot(self.layer1.T, self.del_3)
			dw1 = np.dot(self.inputs.T, self.del_2)
			self.weights2 -= learning_rate * dw2
			self.weights1 -= learning_rate * dw1
			costs.append(self.costFunction())

			if i%100 == 0:
				print("Cost={} at iteration={}".format(costs[-1], i))

		return costs

	def predict(self, x_test):
		l1 = self.sigmoid(np.dot(x_test, self.weights1))
		return self.sigmoid(np.dot(l1, self.weights2))

def generate_data():
	'''
	Just to generate data for xor operation
	'''
	x = []
	y = []
	for x1 in range(2):
		for x2 in range(2):
			for x3 in range(2):
				for x4 in range(2):
					x.append([x1, x2, x3, x4])
					y.append(x1 ^ x2 ^ x3 ^ x4)
	return x, y

def main():

	x, y = generate_data()
	x, y = np.array(x), np.array(y)
	y = np.reshape(y, (y.size, 1))
	model1 = NeuralNet(x, y)
	costs = model1.train(learning_rate=0.1, iterations=10000)
	y_test = model1.predict(np.array([[1, 0, 1, 1], [1, 0, 1, 0]]))
	print(y_test)
	fig = plt.figure(figsize = (10, 10))
	plt.plot(list(range(10000)), costs)
	plt.xlabel("Iteration")
	plt.ylabel("Cost")
	plt.show()


if __name__ == '__main__':
	main()