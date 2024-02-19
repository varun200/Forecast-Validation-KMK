import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import plotly.graph_objs as go
import os
from scipy.stats import t
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import plotly.express as px
from openpyxl import Workbook
from openpyxl.drawing.image import Image

warnings.filterwarnings("ignore")

# Function to calculate decalendarized sales
def calculate_decalendarized_sales(actual_sales, calendarization_values):
    return actual_sales / calendarization_values

# Function to calculate calendarized sales
def calculate_calendarized_sales(actual_sales, calendarization_values):
    return actual_sales * calendarization_values

# Function for linear forecast decalendarized
def linear_forecast_decalendarized(data, start_date, end_date, product_column):
    X = data[['DataPeriodNumber']]
    y = data[product_column]
    model = LinearRegression().fit(X, y)
    
    # Determine the frequency of the data
    data_period_diff = data['Dataperiod'].diff().dt.days.mean()
    if data_period_diff > 28:
        freq = 'M'  # Assuming monthly data if average difference > 28 days
    else:
        most_common_day = pd.to_datetime(data['Dataperiod']).dt.day_name().mode()[0]
        freq = f'W-{most_common_day[:3].upper()}'  # Construct weekly frequency string
    
    # Generate dates between start_date and end_date with the determined frequency
    future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    future_dataperiods = pd.DataFrame({'DataPeriodNumber': range(data['DataPeriodNumber'].max() + 1, data['DataPeriodNumber'].max() + 1 + len(future_dates))})
    future_predictions_linear = model.predict(future_dataperiods[['DataPeriodNumber']])

    return pd.DataFrame({'Dataperiod': future_dates,
                         'Linear Forecast': future_predictions_linear})

# Function for exponential forecast decalendarized
def exponential_forecast_decalendarized(data, start_date, end_date, product_column, alpha, seasonal_periods=None, trend=None, seasonal=None):
    X = data[['DataPeriodNumber']]
    y = data[product_column]
    
    # Fit exponential smoothing model
    model = ExponentialSmoothing(y, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal)
    fitted_model = model.fit()
    
    # Determine the frequency of the data
    data_period_diff = data['Dataperiod'].diff().dt.days.mean()
    if data_period_diff > 28:
        freq = 'M'  # Assuming monthly data if average difference > 28 days
    else:
        most_common_day = pd.to_datetime(data['Dataperiod']).dt.day_name().mode()[0]
        freq = f'W-{most_common_day[:3].upper()}'  # Construct weekly frequency string
    
    # Generate dates between start_date and end_date with the determined frequency
    future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    future_dataperiods = pd.DataFrame({'DataPeriodNumber': range(data['DataPeriodNumber'].max() + 1, data['DataPeriodNumber'].max() + 1 + len(future_dates))})
    future_predictions_exponential = fitted_model.forecast(len(future_dates))
    
    # Calculate standard deviation of errors
    residuals = y - fitted_model.fittedvalues
    std_error = np.std(residuals, ddof=1)  # Sample standard deviation
    
    # Calculate degrees of freedom
    n = len(y)
    if trend is not None:
        p = 2 if seasonal is not None else 1
    else:
        p = 1 if seasonal is not None else 0
    df = n - p
    
    # Calculate t-statistic
    t_stat = t.ppf(1 - alpha / 2, df)
    
    # Calculate confidence intervals
    lower_bound = future_predictions_exponential - t_stat * std_error * np.sqrt(1 + 1 / n)
    upper_bound = future_predictions_exponential + t_stat * std_error * np.sqrt(1 + 1 / n)
    
    # Combine forecast and confidence intervals into a DataFrame
    forecast_df = pd.DataFrame({'Dataperiod': future_dates,
                                'Exponential Forecast': future_predictions_exponential,
                                'Lower CI': lower_bound,
                                'Upper CI': upper_bound})
    
    return forecast_df

# Function to perform outlier normalization
def normalize_outliers(df1):
    outliers_found = False
    for column in df1.columns[1:]:
        if isinstance(column, str) and 'sales' in column.lower() and 'whitespacesales' not in column.lower():
            product_sales = df1[column]
            z_scores = (product_sales - product_sales.mean()) / product_sales.std()

            # Identify outliers using a threshold (e.g., z-score > 3 or z-score < -3)
            outliers = z_scores[(z_scores > 3) | (z_scores < -3)]

            if outliers.any():  
                st.write(f"Outliers detected in {column}:")
                st.write(df1.loc[outliers.index])

                # Normalize outliers (replace them with the mean)
                normalize_option = st.radio(f"Do you want to normalize outliers in {column}?", ('No', 'Yes'), key=f"normalize_{column}")
                
                # Check if user selected "Yes"
                if normalize_option == 'Yes':
                    df1.loc[outliers.index, column] = product_sales.mean()
                    st.write(f"Outliers in {column} normalized.")
                    outliers_found = True
                else:
                    st.write(f"Outliers in {column} kept as they are.")
    
    if outliers_found:
        # Overwrite the session state variable with the updated DataFrame if outliers were found and normalized
        st.session_state.uploaded_file = df1
        st.success("Outliers normalized and DataFrame updated.")
    else:
        st.success("No outliers found 😊")


# Function to check for duplicates and remove them
def remove_duplicates(df1):
    duplicates_found = False
    # Check for duplicates in the DataFrame
    duplicate_rows = df1[df1.duplicated()]
    if not duplicate_rows.empty:
        duplicates_found = True
        st.write("Duplicates found:")
        st.write(duplicate_rows)
        # Remove duplicates
        df1.drop_duplicates(inplace=True)
    return duplicates_found


# Function to get section parameters from user
def get_section_parameters():
    section_parameters = {}
    section_parameters['Baseline Start Period'] = st.date_input("Enter Baseline Start date:", value=pd.to_datetime('today') - pd.DateOffset(months=1))
    section_parameters['Baseline End Period'] = st.date_input("Enter Baseline End date:", value=pd.to_datetime('today'))
    section_parameters['Confidence Interval'] = st.slider("Confidence Interval (%)", min_value=0, max_value=100, step=1, value=95)
    section_parameters['New Quarter Forecast'] = st.number_input("Enter Forecast for New Quarter:")
    section_parameters['Previous Quarter Forecast'] = st.number_input("Enter Forecast for Previous Quarter:")
    section_parameters['Start Date'] = st.date_input("Enter Forecast Start date:", value=pd.to_datetime('today'))
    section_parameters['End Date'] = st.date_input("Enter Forecast End date:", value=pd.to_datetime('today') + pd.DateOffset(months=1))
    # Check if the end date is greater than the start date
    if section_parameters['End Date'] <= section_parameters['Start Date']:
        raise st.error("Error: End date must be later than the start date.")
    return section_parameters

def plot_sales_forecast(df1, df2, region):
    df1_region = df1[['Dataperiod', f'{region}']]
    df2_region = df2[df2['Products'].str.startswith(f'{region}')]
    df2_region = df2_region.set_index('Products').T.reset_index()
    df2_region.columns.name = None
    df2_region = df2_region.rename(columns={'index': 'Dataperiod'})
    
    df1_region['Dataperiod'] = pd.to_datetime(df1_region['Dataperiod'])
    
    df1_region = df1_region[df1_region['Dataperiod'] < df2_region['Dataperiod'].min()]
    df = pd.merge(df1_region, df2_region, on='Dataperiod', how='outer')
    df['Dataperiod'] = pd.to_datetime(df['Dataperiod'])
    fig = px.line(df, x='Dataperiod', y=[f'{region}', *[col for col in df.columns if col != 'Dataperiod' and col != f'{region}']],
                labels={'variable': 'Forecast Type', 'value': 'Sales'},
                title=f'{region} Forecasts')
    st.plotly_chart(fig)


# Main part of the app
st.title("Forecasting App 📈")

# Sidebar navigation
st.sidebar.image("https://s3.amazonaws.com/resumator/customer_20180530150722_IWPNYHFT1IRE4DJX/logos/20230522150306_KMK_Logo_New_2.png", width=200)
st.sidebar.title("Navigation")
menu = ["Upload files", "Data Validation", "Forecast Parameters","Results"]
choice = st.sidebar.button(menu[0], key=menu[0])
if choice:
    st.session_state.tab_selection = menu[0]
choice = st.sidebar.button(menu[1], key=menu[1])
if choice:
    st.session_state.tab_selection = menu[1]
choice = st.sidebar.button(menu[2], key=menu[2])
if choice:
    st.session_state.tab_selection = menu[2]
choice = st.sidebar.button(menu[3], key=menu[3])
if choice:
    st.session_state.tab_selection = menu[3]

# Upload Excel file
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "tab_selection" not in st.session_state or st.session_state.tab_selection == "Upload files":
    st.subheader("Upload Excel file")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df1 = pd.read_excel(uploaded_file, sheet_name=0)
        df2 = pd.read_excel(uploaded_file, sheet_name=1)
        df1 = df1.T
        df1.columns = df1.iloc[0]
        df1 = df1[1:]
        df1.reset_index(inplace=True)
        df1.rename(columns={'index':'Dataperiod'}, inplace=True)

        df2 = df2.T
        df2.columns = df2.iloc[0]
        df2 = df2[1:]
        df2.reset_index(inplace=True)
        df2.rename(columns={'index':'Dataperiod'}, inplace=True)

        df1 = df1.rename_axis(index=None, columns=None)
        df2 = df2.rename_axis(index=None, columns=None)
        df1.iloc[:, 1:] = df1.iloc[:, 1:].astype(float)
        df2.iloc[:, 1:] = df2.iloc[:, 1:].astype(float)
        st.write("Sales Data", df1)
        st.write("")
        st.session_state.file_uploaded = True
        st.session_state.uploaded_file = df1
        st.session_state.uploaded_file1 = df2

        st.success("File has been uploaded.") 
        if st.button("Generate Profiling Report"):
            st.subheader("Data Report")
            report=ProfileReport(df1,title='Forecast Dataset')
            st_profile_report(report)
# Data Validation
elif st.session_state.tab_selection == "Data Validation":
    st.subheader("Data Validation")
    if not st.session_state.file_uploaded:
        st.warning("Please upload the Excel file first.")
    else:
        if st.button("Validate Data"):
            df1 = st.session_state.uploaded_file
            df2 = st.session_state.uploaded_file1
            # Check for null values and zeros in df1
            nulls_df1 = df1.isnull().any().any()
            zeros_df1 = (df1 == 0).any().any()

            # Check for null values and zeros in df2
            nulls_df2 = df2.isnull().any().any()
            zeros_df2 = (df2 == 0).any().any()

            # Check if either df1 or df2 has null values or zeros
            if nulls_df1 or zeros_df1:
                st.error("Error: Null values or zeros found in Sales File. Please fix the error and reupload the file.")
            elif nulls_df2 or zeros_df2:
                st.error("Error: Null values or zeros found in Calenderization. Please fix the error and reupload the file.")
            else:
                st.success("No null values and zeros found in Files.")

            duplicates_found = remove_duplicates(df1)
            if duplicates_found:
                st.success("Duplicates removed.")
            else:
                st.success("No duplicates found.")

            normalize_outliers(df1)
            
            st.success("Data Validation complete!")
            st.session_state.uploaded_file = df1
            st.session_state.uploaded_file1 = df2
            

# Forecasting
elif st.session_state.tab_selection == "Forecast Parameters":
    st.subheader("Forecast Parameters")
    if not st.session_state.file_uploaded:
        st.warning("Please upload the Excel file first.")
    else:
        st.session_state.forecast_parameters = get_section_parameters()
        st.write("Section Parameters:", st.session_state.forecast_parameters)
        st.success("Forecast parameters selected!")

elif st.session_state.tab_selection == "Results":
    st.subheader("Results")
    if not st.session_state.file_uploaded:
        st.warning("Please upload the Excel file first.")
    else:
        df1 = st.session_state.uploaded_file
        df2 = st.session_state.uploaded_file1
        section_parameters = st.session_state.forecast_parameters
        
        # Your results code
        merged_data = pd.merge(df1, df2, on='Dataperiod')
        merged_data['Dataperiod'] = pd.to_datetime(merged_data['Dataperiod']).dt.date

        merged_data = merged_data.sort_values(by='Dataperiod', ascending=False)
        merged_data['DataPeriodNumber'] = range(1, len(merged_data) + 1)
        merged_data = merged_data[['DataPeriodNumber'] + [col for col in merged_data.columns if col != 'DataPeriodNumber']]
        merged_data = merged_data.sort_values(by='Dataperiod', ascending=True)
        forecast_results = pd.DataFrame()
        for product_column in merged_data.columns[2:]:
            if 'sales' in product_column.lower() and 'whitespacesales' not in product_column.lower():
                selected_data = merged_data[(merged_data['Dataperiod'] <= section_parameters['Baseline End Period']) & (merged_data['Dataperiod'] >= section_parameters['Baseline Start Period'])]

                selected_data['DecalendarizedSales'] = calculate_decalendarized_sales(selected_data[product_column], selected_data['Calenderization'])

                linear_forecast_result = linear_forecast_decalendarized(selected_data, section_parameters['Start Date'], section_parameters['End Date'], 'DecalendarizedSales')

                exponential_forecast_result = exponential_forecast_decalendarized(selected_data, section_parameters['Start Date'], section_parameters['End Date'], 'DecalendarizedSales', 1 - section_parameters['Confidence Interval'] / 100)

                lf = linear_forecast_result.merge(df2, on='Dataperiod')
                ef = exponential_forecast_result.merge(df2, on='Dataperiod')

                lf['CalendarizedSales'] = calculate_calendarized_sales(lf['Linear Forecast'], df2['Calenderization'])
                ef['CalendarizedSales'] = calculate_calendarized_sales(ef['Exponential Forecast'], df2['Calenderization'])
                ef['UpperLimits'] = calculate_calendarized_sales(ef['Upper CI'], df2['Calenderization'])
                ef['LowerLimits'] = calculate_calendarized_sales(ef['Lower CI'], df2['Calenderization'])

                forecast_results[f"{product_column} Linear Forecast"] = lf['CalendarizedSales'].round(2)
                forecast_results[f"{product_column} Exponential Forecast"] = ef['CalendarizedSales'].round(2)
                forecast_results[f"{product_column} Upper Limit"] = ef['UpperLimits'].round(2)
                forecast_results[f"{product_column} Lower Limit"] = ef['LowerLimits'].round(2)

        forecast_results['Date'] = lf['Dataperiod']
        forecast_results.set_index(['Date'], inplace=True)
        forecast_results_transposed = forecast_results.T
        
        
        st.write(forecast_results_transposed)
        # Plot sales forecast for each region
        df = forecast_results_transposed.reset_index()
        df = df.rename_axis(index=None, columns=None)
        df.rename(columns={'index': 'Products'}, inplace=True)
        
        for region in df1.columns[1:]:
            plot_sales_forecast(selected_data, df, region)

        def export_excel():
            with pd.ExcelWriter('forecast_results.xlsx') as writer:
                forecast_results_transposed.to_excel(writer, index=True)
            st.success("Exported forecast results to Excel!")
            # Send file to the user to download
            with open('forecast_results.xlsx', 'rb') as f:
                data = f.read()
            st.download_button(
                label="Download Excel file",
                data=data,
                file_name='forecast_results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
            
        st.markdown("---")
        st.write("Export the transposed forecast results to Excel:")
        if st.button("Export to Excel"):
            export_excel()