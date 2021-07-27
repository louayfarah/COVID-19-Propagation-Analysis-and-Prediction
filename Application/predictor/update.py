# -*- coding: utf-8 -*-

import requests                      #For downloading the dataset

import pandas as pd                  #For data exploration and preprocessing
import numpy as np                   #For data manipulation
import matplotlib.pyplot as plt      #For data plotting

from sklearn.model_selection import train_test_split #For splitting data
from sklearn.linear_model import LinearRegression    #For creating model
from sklearn.preprocessing import PolynomialFeatures #For fitting features to polynomial regression model
from sklearn.metrics import mean_squared_error       #For calculating MSE
from sklearn.metrics import r2_score                 #For calculating R-squared

import os                      #For system manipulation                       
import pickle                  #For Updating models



"""# Loading Dataset"""

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.dirname(dir_path))

df_url = 'https://covid19.who.int/WHO-COVID-19-global-data.csv'
req = requests.get(df_url)
df_content = req.content
df_file = open('data/WHO-COVID-19-global-data.csv', 'wb')
df_file.write(df_content)
df_file.close()

try:
  print('Loading dataset....')
  print('Note: Dataset link: https://covid19.who.int/WHO-COVID-19-global-data.csv')
  # Load dataset with comma-seperated values named 'WHO-COVID-19-global-data.csv'
  df=pd.read_csv('data/WHO-COVID-19-global-data.csv', sep=',')
except:
  print("ERROR! Check the dataset in 'data' folder.")
  close=input('Enter any character to exit....')  


"""# Preprocessing data"""

print('Preprocessing....')
result=False
l=df.iloc[:,2].value_counts().tolist()
if len(l)>0:
  result = all(elem == l[0] for elem in l)

if result==False:
  close2=input('ERROR! Todays dataset is not officially completed by WHO yet. Try again later\n Enter any char to exit....')
  raise AssertionError


#Replacing NaN values in Namibia country code feature by 'NM'
df.fillna('NM', inplace=True)

regions=len(df.iloc[:,3].value_counts().index)
countries=len(df.iloc[:,2].value_counts().index)
obs=(len(df)//countries)


"""# 1: Covid-19 Global prediction models
"""

print('Training Covid-19 prediction models....')


df_nc=df.iloc[:, [4,0]]

df_cc=df.iloc[:, [5,0]]

df_nd=df.iloc[:, [6,0]]

df_cd=df.iloc[:, [7,0]]


def sumCasesOrDeathes(df):
  i=0
  df1=df.iloc[i:i+obs,:]
  i+=obs
  for country in range(1,countries):
    value=0
    d={}
    for key in range(i,i+obs):
      d[key]=value
      value+=1
    df1=df1+(df.iloc[i:i+obs,:].rename(index=d))
    i+=obs
  return df1


df_nc1=sumCasesOrDeathes(df_nc)
df_nc1=df_nc1.drop('Date_reported', axis = 1)
df_nc1['day_index']=[i for i in range(1,obs+1)]


df_cc1=sumCasesOrDeathes(df_cc)
df_cc1=df_cc1.drop('Date_reported', axis = 1)
df_cc1['day_index']=[i for i in range(1,obs+1)]


df_nd1=sumCasesOrDeathes(df_nd)
df_nd1=df_nd1.drop('Date_reported', axis = 1)
df_nd1['day_index']=[i for i in range(1,obs+1)]


df_cd1=sumCasesOrDeathes(df_cd)
df_cd1=df_cd1.drop('Date_reported', axis = 1)
df_cd1['day_index']=[i for i in range(1,obs+1)]



"""
**Modeling:**

*new cases*
"""

#choosing the target and the feature
x_nc=df_nc1.iloc[:,[1]].values
y_nc=df_nc1.iloc[:,[0]].values

#splitting the data
x_train_nc,x_test_nc,y_train_nc,y_test_nc=train_test_split(x_nc,y_nc,test_size=0.30,random_state=40)

#fitting the feature
poly_nc=PolynomialFeatures(degree=6)

x_train_nc_fit=poly_nc.fit_transform(x_train_nc)
x_test_nc_fit=poly_nc.fit_transform(x_test_nc)

#creating the regression model
model_nc=LinearRegression()
model_nc.fit(x_train_nc_fit, y_train_nc)



"""*Cumulative cases*"""

#choosing the target and the feature
x_cc=df_cc1.iloc[:,[1]].values
y_cc=df_cc1.iloc[:,[0]].values

#splitting the data
x_train_cc,x_test_cc,y_train_cc,y_test_cc=train_test_split(x_cc,y_cc,test_size=0.30,random_state=40)

#fitting the feature
poly_cc=PolynomialFeatures(degree=2)

x_train_cc_fit=poly_cc.fit_transform(x_train_cc)
x_test_cc_fit=poly_cc.fit_transform(x_test_cc)

#creating the regression model
model_cc=LinearRegression()
model_cc.fit(x_train_cc_fit, y_train_cc)




"""*new deaths*"""

#choosing the target and the feature
x_nd=df_nd1.iloc[:,[1]].values
y_nd=df_nd1.iloc[:,[0]].values

#splitting the data
x_train_nd,x_test_nd,y_train_nd,y_test_nd=train_test_split(x_nd,y_nd,test_size=0.30,random_state=40)

#fitting the feature
poly_nd=PolynomialFeatures(degree=6)

x_train_nd_fit=poly_nd.fit_transform(x_train_nd)
x_test_nd_fit=poly_nd.fit_transform(x_test_nd)

#creating the regression model
model_nd=LinearRegression()
model_nd.fit(x_train_nd_fit, y_train_nd)




"""*Cumulative deaths*"""

#choosing the target and the feature
x_cd=df_cd1.iloc[:,[1]].values
y_cd=df_cd1.iloc[:,[0]].values

#splitting the data
x_train_cd,x_test_cd,y_train_cd,y_test_cd=train_test_split(x_cd,y_cd,test_size=0.30,random_state=40)

#fitting the feature
poly_cd=PolynomialFeatures(degree=2)

x_train_cd_fit=poly_cd.fit_transform(x_train_cd)
x_test_cd_fit=poly_cd.fit_transform(x_test_cd)

#creating the regression model
model_cd=LinearRegression()
model_cd.fit(x_train_cd_fit, y_train_cd)




"""# 2: Covid-19 Local prediction models in Tunisia
"""

dft=df[df.iloc[:,1]=='TN']

start=dft.index[0]
stop=dft.index[-1]

value_t=0
dt={}
for key in range(start, (stop+1)):
  dt[key]=value_t
  value_t+=1

dft=dft.rename(index=dt)

dft['day_index']=[i for i in range(1,obs+1)]

dft_nc=dft.iloc[:,[4,8]]


dft_cc=dft.iloc[:,[5,8]]


dft_nd=dft.iloc[:,[6,8]]


dft_cd=dft.iloc[:,[7,8]]



"""
**Modeling**

*new cases*
"""


#choosing the target and the feature
xt_nc=dft_nc.iloc[:,[1]].values
yt_nc=dft_nc.iloc[:,[0]].values

#splitting the data
xt_train_nc,xt_test_nc,yt_train_nc,yt_test_nc=train_test_split(xt_nc,yt_nc,test_size=0.30,random_state=40)

#fitting the feature
polyt_nc=PolynomialFeatures(degree=4)

xt_train_nc_fit=polyt_nc.fit_transform(xt_train_nc)
xt_test_nc_fit=polyt_nc.fit_transform(xt_test_nc)

#creating the regression model
modelt_nc=LinearRegression()
modelt_nc.fit(xt_train_nc_fit, yt_train_nc)


"""*cumulative cases*"""

#choosing the target and the feature
xt_cc=dft_cc.iloc[:,[1]].values
yt_cc=dft_cc.iloc[:,[0]].values

#splitting the data
xt_train_cc,xt_test_cc,yt_train_cc,yt_test_cc=train_test_split(xt_cc,yt_cc,test_size=0.30,random_state=40)

#fitting the feature
polyt_cc=PolynomialFeatures(degree=2)

xt_train_cc_fit=polyt_cc.fit_transform(xt_train_cc)
xt_test_cc_fit=polyt_cc.fit_transform(xt_test_cc)

#creating the regression model
modelt_cc=LinearRegression()
modelt_cc.fit(xt_train_cc_fit, yt_train_cc)



"""*new deaths*"""

#choosing the target and the feature
xt_nd=dft_nd.iloc[:,[1]].values
yt_nd=dft_nd.iloc[:,[0]].values

#splitting the data
xt_train_nd,xt_test_nd,yt_train_nd,yt_test_nd=train_test_split(xt_nd,yt_nd,test_size=0.30,random_state=40)

#fitting the feature
polyt_nd=PolynomialFeatures(degree=4)

xt_train_nd_fit=polyt_nd.fit_transform(xt_train_nd)
xt_test_nd_fit=polyt_nd.fit_transform(xt_test_nd)

#creating the regression model
modelt_nd=LinearRegression()
modelt_nd.fit(xt_train_nd_fit, yt_train_nd)



"""*cumulative deaths*"""

#choosing the target and the feature
xt_cd=dft_cd.iloc[:,[1]].values
yt_cd=dft_cd.iloc[:,[0]].values

#splitting the data
xt_train_cd,xt_test_cd,yt_train_cd,yt_test_cd=train_test_split(xt_cd,yt_cd,test_size=0.30,random_state=40)

#fitting the feature
polyt_cd=PolynomialFeatures(degree=2)

xt_train_cd_fit=polyt_cd.fit_transform(xt_train_cd)
xt_test_cd_fit=polyt_cd.fit_transform(xt_test_cd)

#creating the regression model
modelt_cd=LinearRegression()
modelt_cd.fit(xt_train_cd_fit, yt_train_cd)



#Saving models

print('Saving models...')

date = df.iloc[-1, 0]
with open('data/models', 'wb') as fic:
  ficpickler=pickle.Pickler(fic)
  ficpickler.dump(date)
  ficpickler.dump(df)
  ficpickler.dump(regions)
  ficpickler.dump(countries)
  ficpickler.dump(obs)
  ficpickler.dump(model_nc)
  ficpickler.dump(poly_nc)
  ficpickler.dump(model_nd)
  ficpickler.dump(poly_nd)
  ficpickler.dump(model_cc)
  ficpickler.dump(poly_cc)
  ficpickler.dump(model_cd)
  ficpickler.dump(poly_cd)
  ficpickler.dump(modelt_nc)
  ficpickler.dump(polyt_nc)
  ficpickler.dump(modelt_nd)
  ficpickler.dump(polyt_nd)
  ficpickler.dump(modelt_cc)
  ficpickler.dump(polyt_cc)
  ficpickler.dump(modelt_cd)
  ficpickler.dump(polyt_cd)


os.system("pause")
