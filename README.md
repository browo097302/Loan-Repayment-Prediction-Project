# Loan-Repayment-Prediction-Project
This project uses DNNs to predict whether a loan will be repayed or defaulted on. The data set includes many different features regarding loan details, credit history, borrower information and  more.  The aim is to use machine learning algorithms to build a robust model that financial institutions could apply for predicting loan default risk. 

 
# Table of Contents
1. Introduction
2. Data Cleaning and EDA
3. Training and Running the DNN
4. Resampling and Rebalancing
5. Hyperparameter Tuning
6. Results

## Introduction
This project leverages machine learning to predict whether a borrower will repay a loan. It explores various techniques such as data cleaning, EDA, model training, resampling, and hyperparameter tuning to achieve optimal performance.

## Data Cleaning and EDA
### Data Cleaning
- Handled missing values.
- Drop unwanted columns
- Get dummies for categorical variables  
- Encoded categorical features.
- Impute new values for missing data and input into dataframe
- Convert time formatted data
- Convert address data 

### EDA
  1. Balance of Target Variable, i.e. proportion of repayed loans to defaults
  2. Distributions of features such as loan amounts, sub_grade, annual income to look for potential skews in the data
  3. Feature relationships - Correlations and Pairplots
  4. Time series analysis of int_rate, loan repayement, loan amount and annual income.
  

## Model Training
Trained an initial deep neural network model on the cleaned dataset with the following architecture:
- Input layer with 68 neurons.
- Hidden layers: 138 neurons, 69 neurons.
- Output layer: 1 neuron.
- Drop out rate 0.5
- Batchsize: 128, epochs: 50

## Resampling and Rebalancing
Addressed class imbalance by:
- Applying resampling techniques (e.g., SMOTE, undersampling).
- Retrained the model on the resampled dataset.


## Hyperparameter Tuning
Performed hyperparameter tuning using GridSearchCV to find the optimal parameters for:
- Number of neurons in each layer.
- Number of layers
- Learning rate.
- Batch size.
- Dropout rate.

## Results
- Initial Model Accuracy: 0.89
- Resampled Model Accuracy: 0.87
- Best Model Accuracy after Hyperparameter Tuning: 0.89
- Final model trained and evaluated on test data 

## Installation and Usage
### Prerequisites
- Python 3.7
- Jupyter Notebook
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, keras, tensorflow, ibmlearn, plotly, datetime
