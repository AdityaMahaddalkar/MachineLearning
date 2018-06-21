import numpy as np

def gradient_descent(x, y, theta, alpha, m, iterations):
    '''
    Inputs: x:numpy array of features, y: numpy array of labels, theta: 2x1 numpy array of initial theta0 and
    theta1 assumptions, alpha: learning rate, m: number of features/lenght of x, iterations: required iterations
    (more iterations = more accurate result)
    Output: theta0 and theta1
    '''

    try:
        if x.size != y.size:
            raise Exception("Unequal feature and label lengths")
        if theta.size != 2:
            raise Exception("Theta array length != 2")
        if alpha == 0:
            raise Exception("Learning Rate = 0")
        if x.size != m:
            raise Exception("Number of features not equal")
    except Exception as e:
        print(e)
    
    m1 = np.array([[1, i] for i in x])
    hypothesis = np.dot(m1, theta)
    loss = hypothesis - y
    cost = np.sum(loss**2)/m
    
    del_J = np.array([np.sum(loss)/m, np.sum(loss*x)/m])

    while iterations > 0:
        iterations -= 1
        del_J[0] = np.sum(loss)/m
        del_J[1] = np.sum(loss*x)/m
        theta = theta - alpha * del_J

        loss = np.dot(m1, theta) - y

        if del_J[0] == 0 and del_J[1] == 0:
            return theta[0], theta[1]

    return theta[0], theta[1]


def multivariate_gradient_descent(x, y, theta, alpha, m, iterations):
    '''
    Input: x: matrix of features grouped by column, y:labels wrt x, theta: initial values of theta, alpha: learning rate,
            m: length of features, iterations: number of iterations required(more iterations = more accurate result)
    Output: optimal theta array
    '''

    try:
        if x[:, 0].size != y.size:
            raise Exception("Unequal feature and label lengths")
        if alpha == 0:
            raise Exception("Learning Rate = 0")
    except Exception as e:
        print(e)



    v = np.ones((1, np.size(x[:, 0])))
    x = np.concatenate((v.T, x), axis=1)
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    J = sum(loss**2) / (2 *m)
    #del_J = np.array([[sum(np.dot((hypothesis - y)), x[:,j])] for j in range(np.size(x[:, 1]))]) #Imporvise this
    del_J =[]
    for j in range(np.size(x[1,:])):
        loss = np.dot(hypothesis - y, x[:, j])
        del_J.append(loss)
    del_J = np.array(del_J)


    #iterations = 100000

    while iterations > 0:
        hypothesis = np.dot(x, theta)
        #del_J = np.array([[sum(np.dot(hypothesis - y, x[:, j]))] for j in range(np.size(x[:, 1]))]) #Improvise this

        theta = theta - alpha * del_J

        del_J =[]
        for j in range(np.size(x[1,:])):
            loss = np.dot(hypothesis - y, x[:, j])
            del_J.append(loss)
        del_J = np.array(del_J)

        iterations -= 1
    
    return theta


#### Test Area (Not useful)
'''
def main():
    x = np.array([[1, 2, 3, 4, 5, 6]])
    x = x.T
    y = np.array([1, 2, 3, 4, 5, 6])
    theta = np.array([0.0, 0.0])
    alpha = 0.005
    m = 5
    t = multivariate_gradient_descent(x, y, theta, alpha, m,1000000)
    print(t)

if __name__ == '__main__':
    main()
'''


