import pandas as pd 
import numpy as np
import seaborn as sns

 #loading my data set
    
data = pd.read_csv('C:\\Users\\Pascal\\Desktop\\DataMining\\pokerhand.csv', encoding= 'unicode_escape')

print('Data before handle the missing value')

print (data)

#Handling the missing value Using two ways
#1. Drop all the rows with null value in the certain column [S4]


data= data[pd.notnull(data['S4'])]
print('Data after handle the missing value by drop all the rows with null value in the certain column [S4]')
print(data) 

#2. filling a missing value with previous ones  
data = data.fillna(method ='pad')
print('Data after handle the missing value by filling it automatically with previous ones ')
print(data) 


#Remove duplicate records 
data = data.drop_duplicates()
print('Data after remove the duplicate row')
print(data) 

# Corrolation
print('Correlation')
corr_matri1 = data.corr()    #to get the relationship between columns
print(corr_matri1)
correlated1 = set()     #initial the set, which we want to put in it the name of columns that correlated 

for i in range (len(corr_matri1.columns)):   #i is number of columns
    for j in range (i) :
        if abs(corr_matri1.iloc[i ,j]) >= 0.8:      #if correlation >= .8 add the column name to the set       
            nomeOfColumn = corr_matri1.columns[i]
            correlated1.add(nomeOfColumn)

print("Correlation rate greater than or equal 0.8 for positive, its size = ") #print the size of correlated
print(len(correlated1))
print(correlated1) #print the correlated
correlated2 = set()        #initial the set, which we want to put in it the name of columns that correlated 

for i in range (len(corr_matri1.columns)):   #i is number of columns
    for j in range (i) :
        if abs(corr_matri1.iloc[i ,j]) <=  -0.8:    #if correlation <= -.8 add the column name to the set
            nomeOfColumn = corr_matri1.columns[i]
            correlated2.add(nomeOfColumn)
len(correlated2)
print("Correlation rate less than or equal -0.8 for negative")  #print the size of correlated
print(len(correlated2))
print(correlated2)    



#Apply discretization

#data['Class'] = pd.cut(x=data['\xa0CLASS'], bins=[0,3,6,9],
 #           labels=["0 to 3","3 to 6","6 to 9"])
#print(data)


#Because the data is big so want to take a sample of it to see the importance feature and to remove the irrelevant attributes
new_data = data.head(1000)

X = new_data.iloc[:,0:10]  #independent columns
y = new_data.iloc[:,-1]    #target column which is class

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)    #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


 # Now drop the columns that irrelevant 

data = data.drop(['S1'], axis = 1) #Delete first culomn S1
data = data.drop(['S2'], axis = 1) #Delete culomn S2
data = data.drop(['S5'], axis = 1) #Delete culomn S5



# Now we want to split our dataset to two parts
# First part for training and its 80% from our data 
# Second part for testing and its 20% from our data 
x = data.iloc[:,0:7]        # X is all the predictive attributes 
y =data.iloc[:, -1]          # Y is the goal attribute
#import the suitable library 
from sklearn.model_selection import train_test_split   

#call method to split it with size .2 of data to test, so .8 of data to train 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2)

#Now print x_train & x_test with its length 
print('x_train')
print(x_train)
print('The length of x_train' ) 
print(len(x_train))
 
print('x_test')
print(x_test)
print('The length of x_test' ) 
print(len(x_test))

#Now print y_train & y_test with its length 
print('y_train')
print(y_train)
print('The length of y_train' ) 
print(len(y_train))

print('y_test')
print(y_test)
print('The length of y_test' ) 
print(len(y_test))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 
knn = KNeighborsClassifier(n_neighbors=3)
 
knn.fit(x_train, y_train)
 
# Predict on dataset which model has not seen before
print(knn.predict(x_test))

#split data attribute and label attribute 
#Because we're using unsupervised 
attributes = data.drop(['\xa0CLASS'], axis=1)
labels = data['\xa0CLASS']

#import and create KMeans object
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(attributes)

#predict the clusters lists for the dataset
y_pred = model.predict(attributes)
print(y_pred)

#evaluation stage
from sklearn import metrics
contingecyMatrix = metrics.cluster.contingency_matrix(labels, y_pred)
print(contingecyMatrix)


#get the association rule
from mlxtend.frequent_patterns import association_rules
rules= association_rules(frequentItemsets, min_threshold= 0.7)
print('Rules\n', rules)

#get association rules by using FP-growth algorithm
import pyfpgrowth

#use FP-growth to get patterns with minimum support = 3
patterns = pyfpgrowth.find_frequent_patterns(data, 3)
print('Patterns\n', patterns)
