import numpy as np  # importing numerical computing package
import pandas as pd  # importing pandas data analysis tool
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression # importing MATLAB-like plotting framework
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # importing data splitter
from sklearn.tree import DecisionTreeClassifier 

 
data =  np.loadtxt(fname = 'labor_exercise_wednesday1.csv', delimiter = ',') # Load dataset with numpy, it contains 10 input and 1 binary variables and doesn't have attributes names

X = data[:,0:10] # Get the 10 input variable
y = data[:,10] # Get the 1 binary variable which is our target var


datframe = pd.DataFrame(data= data, columns= ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10', 'Target']) # Making dataframe and add names for the colum

print(f'Number of record: {data.size}')
print('Number of attributes: ', len(datframe.columns.drop('Target')))
print('Number of classes: ',len(datframe.groupby(by='Target').size()))


grouped_data= datframe.groupby('Target') # Group our dataframe by the target values
grouped_mean = grouped_data.mean() 
std = grouped_data.std()

print(grouped_mean)
print(std)

plt.figure(2)
pd.plotting.parallel_coordinates(datframe,class_column='Target',color=['blue','red']); # Paralell coordinates plotting, colored by target values
#plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=2022) # Split the data set with a 30% test size

class_tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 4) # The firts model we use
class_tree.fit(X_train, y_train) # Give the model the dataset
class_tree_score_train= class_tree.score(X_train,y_train)
class_tree_score_test = class_tree.score(X_test,y_test)

print(class_tree_score_test)


log_reg = LogisticRegression(solver="liblinear")  
log_reg.fit(X_train, y_train)
log_reg_train_score = log_reg.score(X_train, y_train)
log_reg_test_score = log_reg.score(X_test, y_test)

print(log_reg_test_score)

neu_necc = MLPClassifier(hidden_layer_sizes=(1,2), activation="logistic", max_iter=11000)
neu_necc.fit(X_train, y_train)
neu_nec_test_score = neu_necc.score(X_test, y_test)
print(neu_nec_test_score)


# If we look at the scores we see that the Logistic regression is the best model, so we continue with that. 
# We gonna use the confussion matrix to get the errors of the model

log_pred = log_reg.predict(X_test) # we need the predections of the model

confuse_m = metrics.confusion_matrix(y_test, log_pred) # It show the the times when the model made a mistake

