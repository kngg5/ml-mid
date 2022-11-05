import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('weatherAUS.csv')
data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RISK_MM'], axis=1, inplace=True)

data.RainToday = [1 if each == 'Yes' else 0 for each in data.RainToday]
data.RainTomorrow = [1 if each == 'Yes' else 0 for each in data.RainTomorrow]

y = data.RainTomorrow.values
x_data = data.drop('RainTomorrow', axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, nu_of_iteration):
    dimension = x_train.shape[0]
    w, b = initialize_weight_bias(dimension)    
    
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, nu_of_iteration)
    
    
    y_test_predictions = prediction(parameters['weight'], parameters['bias'], x_test) 
    
    
    print('Test accuracy: {}%'.format(100 - np.mean(np.abs(y_test_predictions - y_test))*100))


    logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, nu_of_iteration=400)

    