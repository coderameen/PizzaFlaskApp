import pandas as pd 
#KNN
from sklearn.neighbors import KNeighborsClassifier
#KNeighborsRegressor 
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('pizza.csv')
print(df.head())

#x and y axis 
x = df.iloc[:,:-1]#input
y = df.iloc[:,-1]

#splitting dataset into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

#call the algorithm
model = KNeighborsClassifier(n_neighbors = 3)

#train the model
model.fit(x_train,y_train)

#prediction/test
# pred = model.predict([[25,69]])
# print(pred)

pickle.dump(model,open('model.pkl','wb'))