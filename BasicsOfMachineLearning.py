import numpy as np;  # importing numerical computing package
import pandas as pd;  # importing pandas data analysis tool

 
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

print(grouped_data)
print(std)







