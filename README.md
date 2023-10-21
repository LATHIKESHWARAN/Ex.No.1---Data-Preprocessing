# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
Name : LATHIKESHWARAN J
Reg no : 212222230072
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
df.isnull().sum()
df.duplicated()
df.describe()
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
### The dataset :
![l1](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/bba16df0-8932-4c94-b937-0a0937bdd734)
### Dropping unwanted features :
![l2](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/780f89b6-d425-4764-afed-cca4c97bb5c6)
### Checking for null values :
![l3](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/196aaf8e-ba57-4876-9f50-a44ae9045d1c)
### Checking for duplication :
![l4](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/4db89d96-69eb-4d9d-9e77-365737d70c1d)
### Describing the dataset :
![l6](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/c8db92da-6824-4db5-bd72-786c1cc5bbec)
### Scaling the values :
![l7](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/a1890ab1-f510-4fa6-9962-30a44914b441)
### X Features :
![l8](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/065e9114-89cd-449a-9547-91c861455d3b)
### Y Features :
![l9](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/a74775d3-d2f0-4e3c-ab8d-0d9eeae30957)
### Splitting the training and testing dataset :
![l10](https://github.com/LATHIKESHWARAN/Ex.No.1---Data-Preprocessing/assets/119393556/ffae0ef2-c563-494b-bfdb-62403b13b99e)









## RESULT
Thus we have successfully performed Data preprocessing in a data set downloaded from Kaggle
