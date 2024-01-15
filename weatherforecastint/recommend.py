# import pandas as pd 
# import numpy as np 
# import seaborn
# import sklearn.model_selection
# from sklearn.model_selection import train_test_split
# import pickle

# crop= pd.read_csv("Crop_recommendation.csv")

# crop = crop.sample(frac=1, random_state=None).reset_index(drop=True)

# print(crop)
# # for shape
# print(crop.shape) 
# # statistic percentile
# print(crop.describe) 

# #  for all label values and their counts imported
# z=crop['label'].value_counts()
# print(z) 


# # for graphical visual presentation of a particular column

# # import matplotlib.pyplot as plt 
# # seaborn.displot(crop['P'])
# # plt.show()


# # converts into dictionary
# crop_dict = {'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22}


# # label wala column ka naam ab crop num hai aur map function ne kar diya change
# cr=crop['crop_num']=crop['label'].map(crop_dict)

# # crop['crop_num'].value_counts() print converted the string into int
# print(cr.value_counts())

# # label column hta diya
# crop.drop(['label'],axis=1,inplace=True)
# print(crop.head())

# # giving axis, so X will have 7 col and crop num jayega Y mei
# # X is input data, Y is output variable
# X= crop.drop('crop_num',axis=1)
# Y= crop['crop_num']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)


# # zz=X_train.shape  -->(1760, 7)     80% left earlier it was 2200
# # yy= X_test.shape  -->              2-% to test, rest is in test now
# # print(zz)

# from sklearn.preprocessing import MinMaxScaler
# # for scaling properly, through this algo will be more predictable
# ms = MinMaxScaler()

# X_train = ms.fit_transform(X_train)
# X_test = ms.transform(X_test)
# print(X_train)


# # standard scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# # its in a proper range now.
# sc.fit(X_train)
# X_train = sc.transform(X_train)
# X_test = sc.transform(X_test)
# print(X_train)

# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import ExtraTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import accuracy_score

# # create instances of all models
# models = {
#     'Logistic Regression': LogisticRegression(),
#     'Naive Bayes': GaussianNB(),
#     'Support Vector Machine': SVC(),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(),
#     'Bagging': BaggingClassifier(),
#     'AdaBoost': AdaBoostClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(),
#     'Extra Trees': ExtraTreeClassifier(),
# }


# for name, md in models.items():
#     md.fit(X_train,Y_train)
#     ypred = md.predict(X_test)
    
#     print(f"{name}  with accuracy : {accuracy_score(Y_test,ypred)}")
# # it is a classification problem so-
# # for the best accuracy, we choose rfc as our backend model.

# rfc=RandomForestClassifier(random_state=None)
# rfc.fit(X_train,Y_train)
# ypred=rfc.predict(X_test)
# accuracy_score(Y_test,ypred)

# # function

# def recommendation(N,P,k,temp,Humidity,pH,Rainfall):
#     features = np.array([[N,P,k,temp,Humidity,pH,Rainfall]])
#     transformed_features = ms.fit_transform(features)
#     transformed_features = sc.fit_transform(transformed_features)
#     prediction = rfc.predict(transformed_features).reshape(1,-1)

#     return prediction[0] 
# N = 10
# P = 10
# k = 40
# Temperature = 20.0
# Humidity = 100.0
# pH= 7
# Rainfall = 100

# predict = recommendation(N,P,k,Temperature,Humidity,pH,Rainfall)


# crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

# if predict[0] in crop_dict:
#     crop = crop_dict[predict[0]]
#     print("{} is a best crop to be cultivated ".format(crop))
# else:
#     print("Sorry are not able to recommend a proper crop for this environment")

# pickle.dump(rfc,open('model.pkl','wb'))
# pickle.dump(ms,open('minmaxscaler.pkl','wb'))
# pickle.dump(sc,open('standscaler.pkl','wb'))









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('Crop_recommendation.csv')
df.head()
df.info()

if df['N'].all()>90:
    print(df['N'])

df.isnull().sum()


x = df.drop('label', axis = 1)
y = df['label']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, stratify = y, random_state = 1)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
logistic_acc = accuracy_score(y_test, y_pred)
print("Accuracy of logistic regression is " + str(logistic_acc))

from sklearn.tree import DecisionTreeClassifier
model_2 = DecisionTreeClassifier(criterion='entropy',max_depth = 6, random_state = 2)
model_2.fit(x_train, y_train)
y_pred_2 = model_2.predict(x_test)
decision_acc = accuracy_score(y_test, y_pred_2)
print("Accuracy of decision  tree is " + str(decision_acc))

from sklearn.naive_bayes import GaussianNB
model_3 = GaussianNB()
model_3.fit(x_train, y_train)
y_pred_3 = model_3.predict(x_test)
naive_bayes_acc = accuracy_score(y_test, y_pred_3)
print("Accuracy of naive_bayes is " + str(naive_bayes_acc))

from sklearn.ensemble import RandomForestClassifier
model_4 = RandomForestClassifier(n_estimators = 25, random_state=2)
model_4.fit(x_train.values, y_train.values)
y_pred_4 = model_4.predict(x_test)
random_fore_acc = accuracy_score(y_test, y_pred_4)
print("Accuracy of Random Forest is " + str(random_fore_acc))

import joblib 
file_name = 'banao'
joblib.dump(model_4,'banao')

app = joblib.load('banao')
arr = [[90,42,43,20.879744,82.002744,6.502985,202.935536]]
acc = app.predict(arr)

import pickle
Pkl_Filename = "Pickle_RL_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model_4, file)
with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

Pickled_Model