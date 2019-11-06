import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

def load_data():
    """
        Load cifar10 data from source
        https://www.cs.toronto.edu/~kriz/cifar.html
        :return: Training & Testing Data & labels
        """
    def un_pickle(file):
        with open(file, 'rb') as fo:
            return_dict = pickle.load(fo, encoding='latin1')
        return return_dict
    xTr = np.array([], dtype=np.float64).reshape(0, 3072)
    yTr = np.array([], dtype=np.int64).reshape(0)
    for i in range(1, 6):
        dataDict = un_pickle('cifar-10-batches-py/data_batch_'+str(i))
        xTr = np.concatenate((xTr, dataDict['data']), axis=0)
        yTr = np.concatenate((yTr, dataDict['labels']), axis=0)
    xTe = un_pickle('cifar-10-batches-py/test_batch')['data'].astype(np.float64)
    yTe = np.array(un_pickle('cifar-10-batches-py/test_batch')['labels'], dtype=np.int64)
    return (xTr, yTr), (xTe, yTe)

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
        Reshape a 4D tensor of image data to a grid for easy visualization.
        Inputs:
        - Xs: Data of shape (N, H, W, C)
        - ubound: Output grid will have values scaled to the range [0, ubound]
        - padding: The number of blank pixels between elements of the grid
        """
    (N, H, W, C) = Xs.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

def plot_data(image_data):
    """
        Show some example image data
        :param image_data:
        :return:
        """
    
    image_data = image_data.reshape(-1, 3, 32, 32)
    image_data = image_data.transpose([0, 2, 3, 1])
    grid = visualize_grid(image_data)
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.savefig('Example.png')
    plt.clf()

class SVM (object):
    def __init__ (self, input_dim, output_dim):
        self.w = None
        # TODO:
        # - Generate a random SVM weight matrix to compute loss
        #   with standard Normal Distribution and Standard deviation = 0.01.
        # - Store in self.w variable
        sigma=0.01 #will be used for standard deviation
        self.w= np.random.standard_normal((input_dim,output_dim)) #output batch with standard normal distribution
        self.w= sigma*self.w #return batch with standard deviation of 0.01
    # End of your answer
    
    
    def calLoss (self, x, y, reg):
        """
            SVM loss function
            D: Input dimension.
            C: Number of Classes.
            N: Number of example.
            
            Inputs:
            - x: A numpy array of shape (batchSize, D).
            - y: A numpy array of shape (N,) where value < C.
            - reg: (float) regularization strength.
            
            Returns a tuple of:
            - loss as single float.
            - gradient with respect to weights self.W (dW) with the same shape of self.W.
            """
        loss = 0.0
        d_w = np.zeros_like(self.w)
        # TODO:
        # - Compute the SVM loss and store to loss variable.
        # - Compute gradient and store to d_w variable.
        # - Use L2 regularization
        
        #calculate scores matrix
        scores=np.dot(x,self.w)
        num_train=x.shape[0]
        
        #calculate loss
        scores_y=scores[np.arange(num_train),y] #correct scores matrix
        scores_y=np.matrix(scores_y)
        scores_y=np.transpose(scores_y)
        values=np.maximum(0, scores-scores_y +1)#get maximum of computations
        values[np.arange(num_train),y]=0 #sum correct classes to 0
        loss=np.sum(values)
        loss= loss/num_train
        loss+= reg*np.sum(np.square(self.w))
        
        #calculate gradient dW
        ds=np.zeros_like(values)
        ds[values>0]=1 #return 1 for every instance values array is greater than 0
        values_sum=np.sum(ds, axis=1) #sums row of each item in ds array
        ds[np.arange(num_train),y]=-values_sum.T
        d_w=np.dot(x.T, ds)
        
        d_w =d_w/num_train
        d_w += reg*self.w
        
        # End of your answer
        
        return loss, d_w
    
    def train (self, x, y, lr=1e-3, reg=1e-5, iters=100, batchSize=200, verbose=False):
        """
            Train this Svm classifier using stochastic gradient descent.
            D: Input dimension.
            C: Number of Classes.
            N: Number of example.
            
            Inputs:
            - x: training data of shape (N, D)
            - y: output data of shape (N, ) where value < C
            - lr: (float) learning rate for optimization.
            - reg: (float) regularization strength.
            - iter: (integer) total number of iterations.
            - batchSize: (integer) number of example in each batch running.
            - verbose: (boolean) Print log of loss and training accuracy.
            
            Outputs:
            A list containing the value of the loss at each training iteration.
            """
        
        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iters):
            xBatch = None
            yBatch = None
            # TODO:
            # - Sample batchSize from training data and save to xBatch and yBatch
            # - After sampling xBatch should have shape (batchSize, D)
            #                  yBatch (batchSize, )
            # - Use that sample for gradient decent optimization.
            # - Update the weights using the gradient and the learning rate.
            # - Hint: Use np.random.choice
            batchID = np.random.choice(x.shape[0], batchSize, replace=True)
            xBatch = x[batchID]
            yBatch = y[batchID]
            loss, grad = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)
            self.w += -lr * grad
            # End of your answer
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))
                """
                    Show weight for each class after training
                    """
                if self.w is not None:
                    tmpW = self.w[:-1, :]
                    tmpW = tmpW.reshape(3, 32, 32, 10)
                    tmpW = tmpW.transpose([3, 1, 2, 0])
                    tmpWMin, tmpWMax = np.min(tmpW), np.max(tmpW)
                    wPlot = 255.0 * (tmpW - tmpWMin) / (tmpWMax - tmpWMin)
                    grid = visualize_grid(wPlot)
                    plt.imshow(grid.astype('uint8'))
                    plt.axis('off')
                    plt.gcf().set_size_inches(5, 5)
                    plt.savefig('SVM_'+str(i)+'.png')
                    plt.clf()
    return lossHistory

def predict (self, x,):
    """
        Predict the y output.
        
        Inputs:
        - x: training data of shape (N, D)
        
        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
            yPred = np.zeros(x.shape[0])
                # TODO:
                # -  Store the predict output in yPred
                scores=np.dot(x,self.w)  #generate scores matrix with probabilty score of each class
                    yPred=np.argmax(scores, axis=1)#returns index of maximum value for every row
                        
                        
                        # End of your answer
                        return yPred

def calAccuracy (self, x, y):
    acc = 0
        # TODO:
        # -  Calculate accuracy of the predict value in percentage(%) and store to acc variable
        predicted= self.predict(x)#make use of predict function to predict y value
        acc=np.mean(y==predicted)*100 #checks if predicted value is equal to actual value. Computes mean
        #then returns percentage
        
        
        # End of your answer
        return acc

class Softmax (object):
    def __init__ (self, input_dim, output_dim):
        self.w = None
        # TODO:
        # - Generate a random SVM weight matrix to compute loss
        #   with standard Normal Distribution and Standard deviation = 0.01.
        # - Store in self.w variable
        sigma=0.01 #will be used for standard deviation
        self.w= np.random.standard_normal((input_dim,output_dim))
        self.w= 0.01 *self.w
    
    # End of your answer
    
    
    def calLoss (self, x, y, reg):
        """
            SVM loss function
            D: Input dimension.
            C: Number of Classes.
            N: Number of example.
            
            Inputs:
            - x: A numpy array of shape (batchSize, D).
            - y: A numpy array of shape (N,) where value < C.
            - reg: (float) regularization strength.
            
            Returns a tuple of:
            - loss as single float.
            - gradient with respect to weights self.W (dW) with the same shape of self.W.
            """
        loss = 0.0
        d_w = np.zeros_like(self.w)
        # TODO:
        # - Compute the SVM loss and store to loss variable.
        # - Compute gradient and store to d_w variable.
        # - Use L2 regularization
        
        #calculate loss
        num_train=x.shape[0] #number of training units
        score=np.dot(x,self.w) #scores matrix generated from input and weight matrix
        score=score-np.max(score) #scores subtracted from max scores to prevent overfitting
        p=np.exp(score)/np.sum(np.exp(score)) #p matrix
        loss=np.sum(-np.log(p[np.arange(num_train),y])) #sum of -log of scores matrix corresponding to correct values
        loss /= num_train #total loss divided by number of training units
        loss += reg*np.sum(self.w *self.w) #l2 regularization used for weight penalty
        
        
        #calculate d_w
        ds=np.zeros_like(p) #ds array in the shape of p(probability matrix)
        ds[np.arange(num_train),y]=1 #correct values in ds matrix will be equal to 1
        qq= p- ds  #qq is the difference between probabilty matrix and identity like matrix ds
        d_w=np.dot(x.T, qq) #d_w is equal product of transpose of x and qq matrix
        
        d_w /= num_train #gradient divided by number of units
        d_w += reg*self.w #we also add regularization to gradient
        
        # End of your answer
        
        return loss, d_w
    
    def train (self, x, y, lr=1e-3, reg=1e-5, iters=100, batchSize=200, verbose=False):
        """
            Train this Svm classifier using stochastic gradient descent.
            D: Input dimension.
            C: Number of Classes.
            N: Number of example.
            
            Inputs:
            - x: training data of shape (N, D)
            - y: output data of shape (N, ) where value < C
            - lr: (float) learning rate for optimization.
            - reg: (float) regularization strength.
            - iter: (integer) total number of iterations.
            - batchSize: (integer) number of example in each batch running.
            - verbose: (boolean) Print log of loss and training accuracy.
            
            Outputs:
            A list containing the value of the loss at each training iteration.
            """
        
        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iters):
            xBatch = None
            yBatch = None
            # TODO:
            # - Sample batchSize from training data and save to xBatch and yBatch
            # - After sampling xBatch should have shape (batchSize, D)
            #                  yBatch (batchSize, )
            # - Use that sample for gradient decent optimization.
            # - Update the weights using the gradient and the learning rate.
            # - Hint: Use np.random.choice
            batchID = np.random.choice(x.shape[0], batchSize, replace=True)
            xBatch = x[batchID]
            yBatch = y[batchID]
            loss, grad = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)
            self.w += -lr * grad
            # End of your answer
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))
                """
                    Show weight for each class after training
                    """
                if self.w is not None:
                    tmpW = self.w[:-1, :]
                    tmpW = tmpW.reshape(3, 32, 32, 10)
                    tmpW = tmpW.transpose([3, 1, 2, 0])
                    tmpWMin, tmpWMax = np.min(tmpW), np.max(tmpW)
                    wPlot = 255.0 * (tmpW - tmpWMin) / (tmpWMax - tmpWMin)
                    grid = visualize_grid(wPlot)
                    plt.imshow(grid.astype('uint8'))
                    plt.axis('off')
                    plt.gcf().set_size_inches(5, 5)
                    plt.savefig('Softmax_'+str(i)+'.png')
                    plt.clf()
    return lossHistory

def predict (self, x,):
    """
        Predict the y output.
        
        Inputs:
        - x: training data of shape (N, D)
        
        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
            yPred = np.zeros(x.shape[0])
                # TODO:
                # -  Store the predict output in yPred
                scores=np.dot(x,self.w) #scores matrix with probability scores for each class
                    yPred=np.argmax(scores, axis=1) #returns index of maximum value for every row
                        
                        # End of your answer
                        return yPred

def calAccuracy (self, x, y):
    acc = 0
        # TODO:
        # -  Calculate accuracy of the predict value in percentage(%) and store to acc variable
        predicted=self.predict(x)      #calls predict function to predict a label given a number of features
        acc=np.mean(y==predicted)*100 #checks whether or not predicted value is equal to actual value
        #computes mean of value then converts the probabilties to a percentage
        
        # End of your answer
        return acc

if __name__ == "__main__":
    """
        Load cifar-10 data and show some samples
        """
    (xTrain, yTrain), (xTest, yTest) = load_data()
    print(xTrain.shape, xTrain.dtype, yTrain.shape, yTrain.dtype)
    print(xTest.shape, xTest.dtype, yTest.shape, yTest.dtype)
    plot_data(xTrain[:25])
    
    """
        Pre-processing image data by subtracting mean value
        """
    mean_image = np.mean(xTrain, axis=0)
    xTrain -= mean_image
    xTest -= mean_image
    
    """
        Combine weight and bias
        """
    xTrain = np.hstack([xTrain, np.ones((xTrain.shape[0], 1))])
    xTest = np.hstack([xTest, np.ones((xTest.shape[0], 1))])
    
    """
        Define SVM classifier Model
        """
    classifier = SVM(xTrain.shape[1], 10)
    
    """
        Training SVM classifier
        """
    startTime = time.time()
    classifier.train(xTrain, yTrain, lr=1e-7, reg=5e4, iters=2001 ,verbose=True)
    print ('Training time: {0}'.format(time.time() - startTime))
    
    """
        Calculate accuracy (Should get around this)
        Training acc:   36.35%
        Testing acc:    35.58%
        """
    print ('Training acc:   {0}%'.format(classifier.calAccuracy(xTrain, yTrain)))
    print ('Testing acc:    {0}%'.format(classifier.calAccuracy(xTest, yTest)))
    
    """
        Define Softmax classifier Model
        """
    classifier = Softmax(xTrain.shape[1], 10)
    
    """
        Training Softmax classifier
        """
    startTime = time.time()
    classifier.train(xTrain, yTrain, lr=1e-7, reg=5e4, iters=1001, verbose=True)
    print('Training time: {0}'.format(time.time() - startTime))
    
    """
        Calculate accuracy (Should get around this)
        Training acc:   30.00%
        Testing acc:    30.00%
        """
    print('Training acc:   {0}%'.format(classifier.calAccuracy(xTrain, yTrain)))
    print('Testing acc:    {0}%'.format(classifier.calAccuracy(xTest, yTest)))
