import pandas as pd
import numpy as np
import datetime
import matplotlib.dates as mdates
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

avg_days_yr = 365

#TODO: This is currently based on heuristics using the OLS regression slope. This needs to be replaced with a more sophisticated change likelihood model
def fit_ols_regression(df, y="patch_emb_pca_1"): #'cosine_sim'
    # Assuming df is your DataFrame with 'Date' and 'Value' columns
    df['date'] = pd.to_datetime(df['date'])
    #df['days'] = (df['date'] - df['date'].min()).dt.days
    df['unix_date'] = df['date'].astype(np.int64) // 10**9
    # Add a constant to the independent variable for OLS intercept
    X = sm.add_constant(df['unix_date'])
    y = df[y]

    # Fit the OLS model
    model = sm.OLS(y, X)
    results = model.fit()

    intercept, gradient = results.params
    # Fit the linear trendline
    #z = np.polyfit(df['days'], df['cosine_sim'], 1)
    return abs(gradient)

def detect_outliers_by_year(df, column, year_column='year', threshold=2):
    """
    Detects outliers in each year separately based on a specified number of standard deviations.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column (str): The column name for which to detect outliers.
    year_column (str): The column name for the year.
    threshold (float): The number of standard deviations to use for outlier detection.

    Returns:
    DataFrame: A DataFrame with outliers removed.
    """
    def detect_outliers(group):
        mean = group[column].mean()
        std_dev = group[column].std()
        return group[(group[column] < mean - threshold * std_dev) | (group[column] > mean + threshold * std_dev)]

    # Group by year and apply outlier detection
    outliers = df.groupby(year_column).apply(detect_outliers).reset_index(drop=True)
    
    return outliers["date"] 

def fit_harmonic_regression(data, outliers, date_col="date", y_col="cosine_sim_first_obs", 
                                        baseline_start_date=datetime.datetime(2018,1,1), monitoring_start_date=datetime.datetime(2019,1,1), 
                                        deg=3,reg=0.001):

    data[date_col] = pd.to_datetime(data[date_col])
    data["date_numerical"] = data[date_col].apply(lambda x:  mdates.date2num(x))
    t_full= data["date_numerical"]
    y_full = data[y_col]
    
    # Convert datetime to numerical format
    t_fitting=data[(~data["date"].isin(outliers))&(data[date_col]>=baseline_start_date)&(data[date_col]<monitoring_start_date)]["date_numerical"]
    y_fitting=data[(~data["date"].isin(outliers))&(data[date_col]>=baseline_start_date)&(data[date_col]<monitoring_start_date)][y_col]
    
    # Create design matrix with polynomial features
    w = 2 * np.pi / avg_days_yr
    poly = PolynomialFeatures(deg)
    X_fitting = poly.fit_transform(np.column_stack((np.sin(w*t_fitting), np.cos(w*t_fitting))))
    X_full = poly.fit_transform(np.column_stack((np.sin(w*t_full), np.cos(w*t_full))))

    # Fit Lasso regression
    lasso_model = Lasso(alpha=reg)  # Adjust alpha for regularization strength
    lasso_model.fit(X_fitting, y_fitting)
    
    # Predict values
    y_fit = lasso_model.predict(X_full)
    # Calculate absolute deviation
    absolute_deviation = np.abs(y_full - y_fit)
    # Calculate percentage deviation
    percentage_deviation = (absolute_deviation / np.abs(y_fit)) * 100
    
    #collect results
    df=pd.DataFrame()
    df["date"] = data[date_col]
    df["date_numerical"] = data["date_numerical"] 
    df[f"{y_col}_true"] = data[y_col]
    df[f"{y_col}_pred"] = y_fit
    df[f"{y_col}_abs_error"] = absolute_deviation
    df[f"{y_col}_perc_error"] = percentage_deviation
    df["year"] = df["date"].apply(lambda x: x.year)
    df["month"] = df["date"].apply(lambda x: x.month)
    df["year_month"] = df.apply(lambda x: "{}_{}".format(str(x.year),str(x.month)),axis=1)
    
    return lasso_model,poly, df