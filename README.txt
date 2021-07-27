----------------------------------------------------------------------------------------
-------------------------COVID-19 PROPAGATION PREDICTOR---------------------------------
------------------------------------(LOUAY FARAH)---------------------------------------



---------------------------------DESCRIPTION--------------------------------------------

-COVID-19 global and local propagation analysis through preprocessing WHO official datasets, 
exploring data, visualizing it, and discovering properties and features correlations.
-Training regression models that predict future deaths and cases, 
both globally and locally (in Tunisia).
-Creating figures that represent both actual and predicted observations.
-Models evaluation and experimentation.
-Creating a python application that uses the models in order to perform Covid-19 propagation 
predictions globally and locally.


-----------------------USED DEVELOPMENT ENVIRONMENT AND TOOLS----------------------------

-Google Colaboratory
-Jupyter Notebook
-Python
-Requests
-Numpy
-Pandas
-Matplotlib
-Scikit learn
-Pickle/Unpickle


--------------------------HOW DOES THE PREDICTOR WORK?-----------------------------------

Given a choice between global (g) and Tunisia (t) and an integer n,
the predictor uses a set of regression models in order to:
>>Predict new cases n days after the day when the last observation was made.
>>Predict Cumulative cases n days after the day when the last observation was made.
>>Predict new deaths n days after the day when the last observation was made.
>>Predict cumulative deaths n days after the day when the last observation was made.

NOTE: In order to get the most precise prediction, n should be between 1 and 30 inclusive.
NOTE: Please enable 'Default light' theme mode for a better experience.
NOTE: Dataset link: https://covid19.who.int/WHO-COVID-19-global-data.csv


-------------------------------HOW TO USE THE PREDICTOR?----------------------------------

1)Run 'predict.py' file, located in 'COVID-19 propagation predictor\Application\predictor' directory


--------------------------------HOW TO UPDATE THE PREDICTOR?-------------------------------

1)Run 'update.py' file, located in 'COVID-19 propagation predictor\Application\predictor' directory	


----------------HOW TO INSTALL PYTHON AND THE REQUIRED LIBRARIES AND MODULES?--------------

1)Install the latest version of python from the official website: 'www.python.org'
2)open the terminal, than type this 4 commands: 
			'pip install pandas'
		        'pip install matplotlib'
		        'pip install numpy'
		        'pip install sklearn'
