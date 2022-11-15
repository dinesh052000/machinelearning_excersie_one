import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
Bank_credit_Assumption = pd.read_csv(r'C:\Users\DINESH\Downloads\bankadditionalfull1603957400921\bank-additional-full.csv')
X = Bank_credit_Assumption.columns.drop("y")
y = Bank_credit_Assumption['y']
model = LogisticRegression()
credit_data_encoded = pd.get_dummies(Bank_credit_Assumption[X])
# Checking the shape of the input data

X_train,X_test,y_train,y_test = train_test_split(credit_data_encoded, y,test_size=0.3,random_state=100)
model.fit(X_train,y_train)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Getting the accuracy on training data
train_accuracy = model.score(X_train,y_train)
print("Train accuracy = ", train_accuracy)
# Getting the accuracy on test data
test_accuracy = model.score(X_test,y_test)
print("Test accuracy = ", test_accuracy)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
# Importing the required function
from sklearn.metrics import confusion_matrix
# Creating a confusion matrix on the training data
train_conf_matrix = confusion_matrix(y_train,train_predictions)
# Converting the train_conf_matrix into a DataFrame for better readability
pd.DataFrame(train_conf_matrix,columns=model.classes_,index=model.classes_)
# Calculating train accuracy from confusion matrix
train_correct_predictions = train_conf_matrix[0][0]+train_conf_matrix[1][1]
train_total_predictions = train_conf_matrix.sum()
# Confusion matrix for the test data
test_conf_matrix = confusion_matrix(y_test,test_predictions)
pd.DataFrame(test_conf_matrix,columns=model.classes_,index=model.classes_)

train_accuracy = train_correct_predictions/train_total_predictions
print("Train Accuracy rate = ",train_accuracy)

# Calculating test accuracy from confusion matrix
test_correct_predictions = test_conf_matrix[0][0]+test_conf_matrix[1][1]
total_predictions = test_conf_matrix.sum()

test_accuracy = test_correct_predictions/total_predictions
print("Train Accuracy rate =",test_accuracy)

from sklearn.metrics import classification_report
# Generating the report and printing the same
print(classification_report(y_test,test_predictions))






