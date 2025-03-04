```python
!pip install pandas numpy matplotlib seaborn statsmodels pmdarima

```

    Requirement already satisfied: pandas in c:\users\poorn\anaconda3\lib\site-packages (1.2.4)
    Requirement already satisfied: numpy in c:\users\poorn\anaconda3\lib\site-packages (1.22.4)
    Requirement already satisfied: matplotlib in c:\users\poorn\anaconda3\lib\site-packages (3.3.4)
    Requirement already satisfied: seaborn in c:\users\poorn\anaconda3\lib\site-packages (0.11.1)
    Requirement already satisfied: statsmodels in c:\users\poorn\anaconda3\lib\site-packages (0.14.1)
    Requirement already satisfied: pmdarima in c:\users\poorn\anaconda3\lib\site-packages (2.0.4)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\poorn\anaconda3\lib\site-packages (from matplotlib) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\poorn\anaconda3\lib\site-packages (from matplotlib) (2.4.7)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\poorn\anaconda3\lib\site-packages (from matplotlib) (8.2.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\poorn\anaconda3\lib\site-packages (from matplotlib) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\poorn\anaconda3\lib\site-packages (from matplotlib) (1.3.1)
    Requirement already satisfied: six in c:\users\poorn\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    Requirement already satisfied: pytz>=2017.3 in c:\users\poorn\anaconda3\lib\site-packages (from pandas) (2021.1)
    Requirement already satisfied: urllib3 in c:\users\poorn\anaconda3\lib\site-packages (from pmdarima) (1.26.4)
    Requirement already satisfied: setuptools!=50.0.0,>=38.6.0 in c:\users\poorn\anaconda3\lib\site-packages (from pmdarima) (52.0.0.post20210125)
    Requirement already satisfied: packaging>=17.1 in c:\users\poorn\anaconda3\lib\site-packages (from pmdarima) (24.2)
    Requirement already satisfied: Cython!=0.29.18,!=0.29.31,>=0.29 in c:\users\poorn\anaconda3\lib\site-packages (from pmdarima) (0.29.23)
    Requirement already satisfied: scipy>=1.3.2 in c:\users\poorn\anaconda3\lib\site-packages (from pmdarima) (1.6.2)
    Requirement already satisfied: scikit-learn>=0.22 in c:\users\poorn\anaconda3\lib\site-packages (from pmdarima) (0.24.1)
    Requirement already satisfied: joblib>=0.11 in c:\users\poorn\anaconda3\lib\site-packages (from pmdarima) (1.0.1)
    Requirement already satisfied: patsy>=0.5.4 in c:\users\poorn\anaconda3\lib\site-packages (from statsmodels) (1.0.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\poorn\anaconda3\lib\site-packages (from scikit-learn>=0.22->pmdarima) (2.1.0)
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

```


```python
# Load dataset
df = pd.read_csv(r"C:\Users\poorn\Downloads\Sales_Data.csv", parse_dates=["Date"], index_col="Date")

# Display first few rows
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Transaction ID</th>
      <th>Customer ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Product Category</th>
      <th>Quantity</th>
      <th>Price per Unit</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-11-24</th>
      <td>1</td>
      <td>CUST001</td>
      <td>Male</td>
      <td>34</td>
      <td>Beauty</td>
      <td>3</td>
      <td>50</td>
      <td>150</td>
    </tr>
    <tr>
      <th>2023-02-27</th>
      <td>2</td>
      <td>CUST002</td>
      <td>Female</td>
      <td>26</td>
      <td>Clothing</td>
      <td>2</td>
      <td>500</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>2023-01-13</th>
      <td>3</td>
      <td>CUST003</td>
      <td>Male</td>
      <td>50</td>
      <td>Electronics</td>
      <td>1</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2023-05-21</th>
      <td>4</td>
      <td>CUST004</td>
      <td>Male</td>
      <td>37</td>
      <td>Clothing</td>
      <td>1</td>
      <td>500</td>
      <td>500</td>
    </tr>
    <tr>
      <th>2023-06-05</th>
      <td>5</td>
      <td>CUST005</td>
      <td>Male</td>
      <td>30</td>
      <td>Beauty</td>
      <td>2</td>
      <td>50</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.columns)


```

    Index(['Transaction ID', 'Customer ID', 'Gender', 'Age', 'Product Category',
           'Quantity', 'Price per Unit', 'Sales'],
          dtype='object')
    


```python
df.rename(columns=lambda x: x.strip(), inplace=True)  # Remove extra spaces
df.rename(columns={'date': 'Date', 'DATE': 'Date'}, inplace=True)  # Fix case issues

```


```python
df['Sales'].plot(marker='o', linestyle='-', figsize=(12,5), color='blue', label='Sales')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.show()

```


    
![png](output_5_0.png)
    



```python
def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("Data is stationary")
    else:
        print("Data is NOT stationary")

# Check stationarity
adf_test(df['Sales'])

```

    ADF Statistic: -31.776013873379462
    p-value: 0.0
    Data is stationary
    


```python
fig, ax = plt.subplots(1, 2, figsize=(12,5))

plot_acf(df['Sales'].dropna(), ax=ax[0])
plot_pacf(df['Sales'].dropna(), ax=ax[1])

plt.show()

```


    
![png](output_7_0.png)
    



```python
auto_model = auto_arima(df['Sales'], seasonal=False, trace=True, suppress_warnings=True)
auto_model.summary()

```

    Performing stepwise search to minimize aic
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=1.51 sec
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=16180.713, Time=0.05 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=15894.034, Time=0.09 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.36 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=16178.714, Time=0.12 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=15782.854, Time=0.16 sec
     ARIMA(3,1,0)(0,0,0)[0] intercept   : AIC=15716.209, Time=0.17 sec
     ARIMA(4,1,0)(0,0,0)[0] intercept   : AIC=15678.467, Time=0.26 sec
     ARIMA(5,1,0)(0,0,0)[0] intercept   : AIC=15655.464, Time=0.36 sec
     ARIMA(5,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=2.20 sec
     ARIMA(4,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=2.03 sec
     ARIMA(5,1,0)(0,0,0)[0]             : AIC=15653.465, Time=0.16 sec
     ARIMA(4,1,0)(0,0,0)[0]             : AIC=15676.469, Time=0.21 sec
     ARIMA(5,1,1)(0,0,0)[0]             : AIC=inf, Time=1.83 sec
     ARIMA(4,1,1)(0,0,0)[0]             : AIC=inf, Time=1.02 sec
    
    Best model:  ARIMA(5,1,0)(0,0,0)[0]          
    Total fit time: 10.835 seconds
    




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>   <td>1000</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(5, 1, 0)</td> <th>  Log Likelihood     </th> <td>-7820.733</td>
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 04 Mar 2025</td> <th>  AIC                </th> <td>15653.465</td>
</tr>
<tr>
  <th>Time:</th>                <td>14:03:26</td>     <th>  BIC                </th> <td>15682.906</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>15664.655</td>
</tr>
<tr>
  <th></th>                      <td> - 1000</td>     <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>   -0.8314</td> <td>    0.032</td> <td>  -26.007</td> <td> 0.000</td> <td>   -0.894</td> <td>   -0.769</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -0.6610</td> <td>    0.040</td> <td>  -16.660</td> <td> 0.000</td> <td>   -0.739</td> <td>   -0.583</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>   -0.4994</td> <td>    0.042</td> <td>  -11.872</td> <td> 0.000</td> <td>   -0.582</td> <td>   -0.417</td>
</tr>
<tr>
  <th>ar.L4</th>  <td>   -0.3230</td> <td>    0.041</td> <td>   -7.955</td> <td> 0.000</td> <td>   -0.403</td> <td>   -0.243</td>
</tr>
<tr>
  <th>ar.L5</th>  <td>   -0.1570</td> <td>    0.032</td> <td>   -4.856</td> <td> 0.000</td> <td>   -0.220</td> <td>   -0.094</td>
</tr>
<tr>
  <th>sigma2</th> <td> 3.699e+05</td> <td> 1.48e+04</td> <td>   25.052</td> <td> 0.000</td> <td> 3.41e+05</td> <td> 3.99e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.28</td> <th>  Jarque-Bera (JB):  </th> <td>198.42</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.60</td> <th>  Prob(JB):          </th>  <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.82</td> <th>  Skew:              </th>  <td>1.06</td> 
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.07</td> <th>  Kurtosis:          </th>  <td>3.55</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
# Set ARIMA order based on auto_arima
p, d, q = 1, 1, 1  # Replace with best values from auto_arima

model = ARIMA(df['Sales'], order=(p, d, q))
model_fit = model.fit()

# Print summary
print(model_fit.summary())

```

    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                  Sales   No. Observations:                 1000
    Model:                 ARIMA(1, 1, 1)   Log Likelihood               -7742.526
    Date:                Tue, 04 Mar 2025   AIC                          15491.052
    Time:                        14:03:33   BIC                          15505.772
    Sample:                             0   HQIC                         15496.647
                                   - 1000                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1         -0.0055      0.033     -0.164      0.870      -0.071       0.060
    ma.L1         -0.9999      0.053    -18.908      0.000      -1.104      -0.896
    sigma2      3.137e+05   2.33e+04     13.482      0.000    2.68e+05    3.59e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):               335.56
    Prob(Q):                              0.94   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.83   Skew:                             1.36
    Prob(H) (two-sided):                  0.08   Kurtosis:                         3.79
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
# Forecast 30 days ahead
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Create a date range for future predictions
future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='D')[1:]

# Convert forecast to DataFrame
forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted_Sales": forecast.values})

# Display forecast table
forecast_df.head()

```

    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Forecasted_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-12-05</td>
      <td>457.650993</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-12-06</td>
      <td>455.808695</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-12-07</td>
      <td>455.818747</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-12-08</td>
      <td>455.818692</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-12-09</td>
      <td>455.818692</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Forecast 30 days ahead
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Create a date range for future predictions
future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='D')[1:]

# Convert forecast to DataFrame
forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted_Sales": forecast.values})

# Display forecast table
forecast_df.head()

```

    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\poorn\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Forecasted_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-12-05</td>
      <td>457.650993</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-12-06</td>
      <td>455.808695</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-12-07</td>
      <td>455.818747</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-12-08</td>
      <td>455.818692</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-12-09</td>
      <td>455.818692</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,5))
plt.plot(df.index, df['Sales'], label="Actual Sales", color="blue")
plt.plot(forecast_df["Date"], forecast_df["Forecasted_Sales"], label="Forecasted Sales", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Actual vs Forecasted Sales")
plt.legend()
plt.show()

```


    
![png](output_12_0.png)
    



```python
# Compare forecasted values with actual last 30 days (if available)
actual = df['Sales'].iloc[-forecast_steps:].values
predicted = forecast.values[:len(actual)]

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f"RMSE: {rmse:.2f}")

# MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(actual, predicted) * 100
print(f"MAPE: {mape:.2f}%")

```

    RMSE: 377.79
    MAPE: 496.30%
    


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r"C:\Users\poorn\Downloads\heart_disease.csv")

# Check for missing values
print(df.isnull().sum())

# Convert gender to numeric
df['gender'] = df['gender'].map({'male': 1, 'female': 0})

# Convert Blood Pressure column
df['systolic_BP'] = df['blood pressure'].apply(lambda x: 120 if x == 'systolic' else 80)  
df.drop(columns=['blood pressure'], inplace=True)  # Drop original column

```

    age               0
    gender            0
    heart disease     0
    blood pressure    0
    cholesterol       0
    dtype: int64
    


```python
scaler = StandardScaler()
df[['age', 'cholesterol', 'systolic_BP']] = scaler.fit_transform(df[['age', 'cholesterol', 'systolic_BP']])

```


```python
# Split data
X = df[['age', 'gender', 'cholesterol', 'systolic_BP']]  # Features
y = df['heart disease']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

```




    LogisticRegression()




```python
# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

```

    Accuracy: 0.5853658536585366
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.45      0.67      0.54        15
               1       0.74      0.54      0.62        26
    
        accuracy                           0.59        41
       macro avg       0.60      0.60      0.58        41
    weighted avg       0.63      0.59      0.59        41
    
    
    Confusion Matrix:
     [[10  5]
     [12 14]]
    


```python

```
