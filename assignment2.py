import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

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
    plt.show(bbox_inches='tight')
    plt.clf()

def pythonSVMLoss(x, y, w, reg):
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
    """
    loss = 0.0
    scores = np.dot(x,w)
    num_class = len(x)
    num_train = len(x[0])

    for i in range(num_class):
        if(i==y): #Check if i == y
            continue
        loss += max(0,(scores[i]-(scores[y])).all() + 1) #Caluclate loss function

    #Transpose w as we did in Numpy
    t_w = [[0 for j in range(len(w))] for u in range(len(w[0]))] #Fill w with 0s
    for i in range(len(w)):
        for j in range(len(w[0])):
            t_w[j][i] = w[i][j] #Transpose

    #Weight matrix
    l2 = np.dot(w, t_w)
    holder = []
    for i in range(len(w)):
        for j in range(len(w[0])):
            holder.append(l2[j][i])
    l2 = math.fsum(holder)
    l2 = l2 / 10

    loss += 0.5 * reg * l2 #Regularlize fucntion
    return(loss)

def numPySVMLoss(x, y, w, reg):
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
    """
    loss = 0.0
    scores = np.dot(x, w) #Gets score for matrix
    num_train = x.shape[0] #
    # Calculates correct score matrix
    correct_scores = scores[np.arange(num_train),y] #Correct scores for images
    correct_scores = np.matrix(correct_scores) #Make into matrix
    correct_scores = np.transpose(correct_scores) #Transpose through matrix of correct scores
    value = np.maximum(0,scores-correct_scores+1) #Calculate loss function
    value[np.arange(num_train),y] = 0 #Set all correct values = 0 so it does not affect our function
    loss = np.sum(value) #Sum up all values
    loss /= num_train #Divice by total number of training

    loss += 0.5 * reg * np.sum(w*w)

    return(loss)



def pythonSoftmaxLoss(x, y, w, reg):
    """
    Softmax loss function
    D: Input dimension.
    C: Number of Classes.
    N: Number of example.

    Inputs:
    - x: A numpy array of shape (batchSize, D).
    - y: A numpy array of shape (N,) where value < C.
    - reg: (float) regularization strength.

    Returns a tuple of:
    - loss as single float.
    """
    loss = 0.0

    num_train = len(x) #Number of training data
    scores = np.dot(x,w) #Get score of data
    sum_exp = 0 #Set = 0, this is to calculate sum of exponential score

    exp_scores = [] #List to store exponential scores
    for i in range(len(scores)):
        for j in range(len(scores)):
                exp_scores.append(math.exp(scores[i][j])) #Append exponential scores if not overflow
                sum_exp+= math.exp(scores[i][j]) #Keep sum count of exponential score

    for i in range(len(exp_scores)):
        exp_scores[i] /= sum_exp #Divide each exponential score by total exponential score as displayed in formula
    loss=math.fsum(exp_scores)

    t_w = [[0 for i in range(len(w))] for j in range(len(w[0]))] #Transpose through matrix as done with Numpy
    for i in range(len(w)):
        for j in range(len(w[0])):
            t_w[j][i] = w[i][j]

    return loss




    # TODO:
    # - Compute the Softmax loss and store to loss variable.
    # - Use L2 regularization
    # - You cannot use any numpy function except np.dot()
    # - You python math library










def numPySoftmaxLoss(x, y, w, reg):
    """
    Softmax loss function
    D: Input dimension.
    C: Number of Classes.
    N: Number of example.

    Inputs:
    - x: A numpy array of shape (batchSize, D).
    - y: A numpy array of shape (N,) where value < C.
    - reg: (float) regularization strength.

    Returns a tuple of:
    - loss as single float.
    """
    loss = 0.0
    # TODO:
    # - Compute the Softmax loss and store to loss variable.
    # - Use L2 regularization
    num_train = x.shape[0]

    scores = np.dot(x,w) #Caluclate scores of images
    exp_score = np.exp(scores - np.max(scores)) #Get exponential scores of images as seen in formula
    correct_score = exp_score[np.arange(num_train), y] #Correct exponential scores to subtract from training exponential scores
    correct_score = np.matrix(correct_score) #Convert to matrix

    loss=correct_score/np.sum(correct_score) #Calculate loss function
    loss = np.mean(loss) #Complete average of loss function

    loss += 0.5* reg * np.sum(w*w)


    return(loss)

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
    Calculate SVM Loss
    """
    weight = np.zeros(shape=(3073, 10))
    loss_0 = numPySVMLoss(xTrain[:10], yTrain[:10], weight, 1)
    loss_1 = pythonSVMLoss(xTrain[:10].tolist(), yTrain[:10].tolist(), weight.tolist(), 1)
    print(loss_0, loss_1)
    weight = 0.01*np.ones(shape=(3073, 10))
    loss_0 = numPySVMLoss(xTrain[:10], yTrain[:10], weight, 1)
    loss_1 = pythonSVMLoss(xTrain[:10].tolist(), yTrain[:10].tolist(), weight.tolist(), 1)
    print(loss_0, loss_1)

    """
    Calculate Softmax Loss
    """
    weight = np.zeros(shape=(3073, 10))
    loss_0 = numPySoftmaxLoss(xTrain[:10], yTrain[:10], weight, 1)
    loss_1 = pythonSoftmaxLoss(xTrain[:10].tolist(), yTrain[:10].tolist(), weight.tolist(), 1)
    print(loss_0, loss_1)
    weight = 0.01*np.ones(shape=(3073, 10))
    loss_0 = numPySoftmaxLoss(xTrain[:10], yTrain[:10], weight, 1)
    loss_1 = pythonSoftmaxLoss(xTrain[:10].tolist(), yTrain[:10].tolist(), weight.tolist(), 1)
    print(loss_0, loss_1)

