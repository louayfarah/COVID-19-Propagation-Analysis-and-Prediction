# -*- coding: utf-8 -*-

# Importing libraries and modules

import pickle
import os


#Loading data

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.dirname(dir_path))

with open('data/models', 'rb') as fic:
  ficunpickler=pickle.Unpickler(fic)
  date=ficunpickler.load()
  df=ficunpickler.load()
  regions=ficunpickler.load()
  countries=ficunpickler.load()
  obs=ficunpickler.load()
  model_nc=ficunpickler.load()
  poly_nc=ficunpickler.load()
  model_nd=ficunpickler.load()
  poly_nd=ficunpickler.load()
  model_cc=ficunpickler.load()
  poly_cc=ficunpickler.load()
  model_cd=ficunpickler.load()
  poly_cd=ficunpickler.load()
  modelt_nc=ficunpickler.load()
  polyt_nc=ficunpickler.load()
  modelt_nd=ficunpickler.load()
  polyt_nd=ficunpickler.load()
  modelt_cc=ficunpickler.load()
  polyt_cc=ficunpickler.load()
  modelt_cd=ficunpickler.load()
  polyt_cd=ficunpickler.load()

#Predictor

print(90*'-')
print('PROPAGATION PREDICTOR OF COVID-19'.center(85))
print(90*'-', end='\n\n\n')

print(38*'-', 'DESCRIPTION', 39*'-', end='\n\n')
print("""Given a choice between global (g) and Tunisia (t) and an integer n,
the predictor uses a set of regression models in order to:
>>Predict new cases n days after the day when the last observation was made.
>>Predict Cumulative cases n days after the day when the last observation was made.
>>Predict new deaths n days after the day when the last observation was made.
>>Predict cumulative deaths n days after the day when the last observation was made.
NOTE: In order to get the most precise prediction, n should be between 1 and 30 inclusive.
""", end= '\n\n\n')



print(35*'-', 'GENERAL INFORMATION', 34*'-', end='\n\n')

print('Dataset:\n {}'.format(df),end='\n')
print('Observations have been made on {0} regions and {1} countries.'.format(regions, countries))
print('Number of observations: {}'.format(obs))
print('Last observation was made on: {0}.'.format(date), end='\n\n')

print(39*'-', 'PREDICTION', 39*'-', end='\n\n')

while True:
  decision=0
  while True:
    area=input("Enter g for global Covid-19 prediction, or t for local Covid-19 prediction in Tunisia: ")
    if area in 'gG':
      decision=1
      break
    elif area in 'tT':
      decision=2
      break
    else:
      print('Please, enter a valid character.')
      continue

  while True:
    try:
      n=int(input("Enter an integer n where n = number of days after last observation day: "))
      break
    except:
      print('Please, enter a valid integer.')
      continue

  print('\n\n')

  if decision==1:
    print('GLOBAL COVID-19 PREDICTION:')
    print('New cases after {0} days from last observation: {1}'.format(n, int(model_nc.predict(poly_nc.fit_transform([[obs+n]])))))
    print('New deaths after {0} days from last observation: {1}'.format(n, int(model_nd.predict(poly_nd.fit_transform([[obs+n]])))))
    print('Cumulative cases after {0} days from last observation: {1} Millions'.format(n, round(int(model_cc.predict(poly_cc.fit_transform([[obs+n]])))/1000000,2)))
    print('Cumulative deaths after {0} days from last observation: {1}K\n\n'.format(n, round(int(model_cd.predict(poly_cd.fit_transform([[obs+n]])))/1000,2)))
  else:
    print('LOCAL COVID-19 PREDICTION (TUNISIA):')
    print('New cases after {0} days from last observation: {1}'.format(n, int(modelt_nc.predict(polyt_nc.fit_transform([[obs+n]])))))
    print('New deaths after {0} days from last observation: {1}'.format(n, int(modelt_nd.predict(polyt_nd.fit_transform([[obs+n]])))))
    print('Cumulative cases after {0} days from last observation: {1}'.format(n, int(modelt_cc.predict(polyt_cc.fit_transform([[obs+n]])))))
    print('Cumulative deaths after {0} days from last observation: {1}\n\n'.format(n, int(modelt_cd.predict(polyt_cd.fit_transform([[obs+n]])))))

  c=input('Enter n to start a new prediction, or e to exit:')
  if c in 'nN':
    continue
  else:
    break
      
  
