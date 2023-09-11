# Developing a Neural Network Regression Model

## AIM:

To develop a neural network regression model for the given dataset.

## THEORY:
Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model:

![](./o1.png)

## DESIGN STEPS:

### STEP 1:

Loading the dataset.

### STEP 2:

Split the dataset into training and testing.

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM:
```
Developed By : Lathika Sunder
Reg No:212221230054
```
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow import keras as k
import matplotlib.pyplot as plt

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('newdata').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'int'})
df = df.astype({'Output':'int'})
df.head()

X=df[['Input']].values
Y=df[['Output']].values

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=33)

scaler=MinMaxScaler()

scaler.fit(x_train)

x_train1=scaler.transform(x_train)
x_train1

ai_brain=Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(optimizer='rmsprop',loss='mse')

ai_brain.fit(x_train1,y_train,epochs=2000)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()

x_test1=scaler.transform(x_test)

ai_brain.evaluate(x_test1,y_test)

x_n1=[[4]]

x_n1_1=scaler.transform(x_n1)

ai_brain.predict(x_n1_1)
```

## Dataset Information:

![](./o2.png)

## OUTPUT:
### Training Loss Vs Iteration Plot:
![](./o3.png)

### Test Data Root Mean Squared Error:
![](./o4.png)

### New Sample Data Prediction:
![](./o5.png)
## RESULT:
Thus a neural network regression model for the given dataset is written and executed successfully.
