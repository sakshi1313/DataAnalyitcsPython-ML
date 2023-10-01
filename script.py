#importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection as ms      
import scipy.optimize as opt                                  #for using minimize function
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_precision_recall_curve


import warnings                                               
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


#-------------------------LOADING THE DATA-----------------------------

df = pd.read_csv('Concrete_Data.csv')

#-------------------- DATA ANALYSIS --------------------
def DataAnalysis(df):
    df.info() # to check the data types of the columns
    # Checking for missing values   
    print(df.isnull().sum())
    cols = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age', 'Concrete_compressive_strength']
    df.columns = cols
    print(df.head(5))
    print(df.describe()) # to check the statistical summary of the data


    # Inferences :
    # Blast_Furnace_Slag and Fly_Ash, Age has wide difference in mean and 50% values,indicating mean > median, so being a left-skewness in data.
    # in cement., min = 102, std = 101, range is 102 - 531, std and min are very close, this implies that the spread of the data is very small or even zero..
    # In col Water , std is < min value, so we have to think whether this data should be considered or not.

#-------------------- DATA VISUALIZATION --------------------
def plot(df):
    sns.distplot(df['Concrete_compressive_strength'])               # to check the distribution of the data
    plt.show()

    print("Skewness = ",df['Concrete_compressive_strength'].skew()) # to check the skewness of the data
    #It is positively, lightly skewed, as the skew value is < 0.5. 

    #plotting the correlation matrix
    corr_DF = df.corr()
    sns.heatmap(corr_DF, annot = True)

    #plotting the pairplot
    sns.pairplot(df)

    sns.scatterplot(y="Concrete_compressive_strength", x="Cement", hue="Water",size="Age", data=df, sizes=(50, 300))
    plt.show()


# -------------------- DATA NORMALIZATION/ Pre-processing --------------------
def normalization(df):
    data_to_normalize = df.iloc[:,:-1]
    # print(data_to_normalize.head())
    scaler = MinMaxScaler()
    data_to_normalize = scaler.fit_transform(data_to_normalize)

    normalized_data = pd.DataFrame(data_to_normalize)

    normalized_data['Concrete_compressive_strength'] = df['Concrete_compressive_strength'] # adding the target column to the normalized data

    print("Noramlized Data :")
    print(normalized_data.head())
    
    #plotting the correlation matrix
    corr_normalised_Data = normalized_data.corr()
    sns.heatmap(corr_normalised_Data, annot = True)
    plt.show()

    return normalized_data


# ---------------------- APPLYING NEURAL NETWORK ----------------------    
def sigmoid(z):                                           #sigmoid function
    return 1 / (1 + np.exp(-z))    
    
def sigmoid_gradient(z):                                      
    '''calculates gradient of sigmoid function'''
    g = sigmoid(z)
    return np.multiply(g, 1 - g)

def random_value_initialization(features, hidden, output):                    
    '''randomly initializes NN parameters'''
    theta1 = (2 * np.random.rand(hidden, features)) - 1
    theta2 = (2 * np.random.rand(output, hidden+1)) - 1
    return (theta1, theta2)

def unroll(mat1, mat2):         
    '''unrolls all Parameter matrices into a single vector for minimize function'''
    return np.concatenate([mat1.ravel(), mat2.ravel()])


def forward_propagate(X, theta1, theta2):        
    ''' forward propagates the input and returns the output'''
    m  = X.shape[0]
    a1 = X
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack(( np.ones(m), a2))
    h = np.dot(a2, theta2.T)
    return h

def cost_function(Theta, X, y, lamda, hidden, output):     
    '''calculates cost''' 
    m, n= X.shape
    flag = n * hidden
    
    theta1 = Theta[0: flag]
    theta2 = Theta[flag:]

    theta1 = theta1.reshape((hidden, n))
    theta2 = theta2.reshape((output, hidden+1))
    
    Y = np.reshape((y.values).T, (m, 1))
    H = forward_propagate(X, theta1, theta2)
    
    Diff = H - Y
    cost = Diff * Diff 
    
    J = (1/(m*2)) * np.sum(cost)  #cost function without regularization


    # regularization
    t1 = theta1[:, 1:]  #removing bias term
    t2 = theta2[:, 1:]
    J = J + (( lamda/(2*m) ) * ( np.sum(t1 * t1) + np.sum(t2 * t2) ))   #adding regularization term
    #print(J)
    return J
    
def back_propagate(Theta, X, y, lamda, hidden, output):              
    '''calculates gradient using backward propagation'''
    m, n= X.shape #
    pos = n * hidden # 
    
    theta1 = Theta[0: pos]
    theta2 = Theta[pos:]

    theta1 = theta1.reshape((hidden, n))
    theta2 = theta2.reshape((output, hidden+1))
    
    Y = np.reshape((y.values).T, (m, 1))
    
    grad1 = np.zeros(theta1.shape) # store the gradients of the cost function with respect to the weights. (dJ/dW1  )
    grad2 = np.zeros(theta2.shape)
    
    for i in range(m):
        x = X[i, :]
        a1 = x.reshape((1,n))
        z2 = np.dot(a1, theta1.T)
        a2 = sigmoid(z2)
        a2 = np.column_stack( (np.ones(a2.shape[0]), a2) )
        z2 = np.column_stack( (np.ones(z2.shape[0]), z2) )
        h = np.dot(a2, theta2.T)
        
        diff = h - Y[i, :]
        
        err1 = np.dot(diff, theta2) * sigmoid_gradient(z2)
        err1_g = np.dot(err1.T, a1)
        err1_g = err1_g[1:, :]
        
        grad1 = grad1 + err1_g
        grad2 = grad2 + (np.dot(diff, a2))
    
    grad1 = grad1/m;
    grad2 = grad2/m;
    
    sum1 = theta1
    sum2 = theta2
    
    sum1[:, 0] = np.zeros((sum1[:, 0]).shape)
    sum2[:, 0] = np.zeros((sum2[:, 0]).shape)
    
    grad1 = grad1 + (lamda/m) * sum1  #adding regularization term
    grad2 = grad2 + (lamda/m) * sum2
    
    vec = unroll(grad1, grad2)
    
    return vec

def accuracy(Theta, X, y, reg_const, hid, op):    
    '''calculates accuracy of the trained NN'''
    m, n= X.shape
    pos = n * hid
    
    theta1 = Theta[0: pos]
    theta2 = Theta[pos:]

    theta1 = theta1.reshape((hid, n))
    theta2 = theta2.reshape((op, hid+1))
    
    Y = np.reshape((y.values).T, (m, 1))
    H= forward_propagate(X, theta1, theta2)

    for i in range(m):
       print("Labled Value: ",Y[i])
       print("Predicted value: ", H[i],"\n")
        
    J = cost_function(Theta, X, y, reg_const, hid, op)
    J = J * m;
    
    val = np.sqrt(J)
    sum_val = np.sum(Y)
    
    return 100 - (val/sum_val)



# 3 Layers with 1 input, 1 hidden, and 1 output layer

def main():

    df = pd.read_csv('Concrete_Data.csv')     #reading data from csv file
    DataAnalysis(df)
    plot(df)

    m, feature = df.shape       #Number of input neurons   
    hidden = 10                 #Number of hidden neurons
    output = 1                  #Number of output neurons

    print("Number of features: ", feature - 1)
    print("Number of training examples: ", m)

    l = 0                       #regularization parameter

    # Normalizing the data- pre-processing 
    normalized_data = normalization(df)
    X = normalized_data.iloc[:, 0 : feature - 1]
    y = normalized_data.Concrete_compressive_strength

    X_biased = np.column_stack(( np.ones(m), X) ) 

    X_train, X_test, y_train, y_test = ms.train_test_split(X_biased, y, test_size = 0.20)      #splitting data in 8:2 ratio for training and testing

    theta1, theta2 = random_value_initialization(feature, hidden, output)      #initialising parameters
    theta = unroll(theta1, theta2)                                             #unrolling parameter matrices into single vector

    print("Training Neural Network...\n\n")

    Result = opt.minimize(fun = cost_function, x0 = theta, args = (X_train, y_train, l, hidden, output), method = 'BFGS', jac = back_propagate) #Minimizing Cost Function 
    opt_theta = Result.x

    print("\nModel Trained:)\n")
    print("Optimum Parameters:", opt_theta, "\n\n")
    print("\nAccuracy obtained: ", accuracy(opt_theta, X_test, y_test, 0, hidden, output ),"%\n\n")   



if __name__ == "__main__":
    main()

