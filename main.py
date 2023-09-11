import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pycaret as py
from pycaret.classification import *
# Load data in various formats, including CSV, Excel
def load_data(file_path):
    
    # Check if data in excel file 
    if file_path.endswith("xlsx") or file_path.endswith("xls"):
        df = pd.read_excel(file_path)
    
    # Check data if in CSV file
    elif file_path.endswith("csv"):
        df = pd.read_csv(file_path)
    
    # Avoid error in file format
    else:
        raise ValueError("unsupported file")
    
    
    return df

def handling_missing_data(df):
    
   
    for column in df.columns:
        pre = df[column].isna().sum() / len(df) * 100
        if pre <= 5:
            df = df.dropna(subset=[column])
        elif 5 < pre < 90:
            if df[column].dtype == "object":
                df[column] = df[column].fillna(df[column].mode()[0])
            elif df[column].dtype in ["int64", "float64"]:
                df[column] = df[column].fillna(df[column].mean())
        else:
            df = df.drop(column, axis=1)
               
        return df

def Train_model(df):
    
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    problem_type = st.selectbox('Choose the Problem Type', ['Regression', 'Classification'])
    if st.button('Run Modelling'):
        if problem_type == 'Regression':
            from pycaret.regression import setup, compare_models,pull, save_model
            setup(df, target=chosen_target, verbose=False)
            st.success('Setup Complete')
            st.subheader('Model Comparison')
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
        elif problem_type == 'Classification':
            from pycaret.classification import setup, compare_models,pull, save_model
            setup(df, target=chosen_target, verbose=False)
            st.success('Setup Complete')
            st.subheader('Model Comparison')
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)

            save_model(best_model, 'best_model')
def visualize_data(data):
    
    plots = []

    for column in data.columns:
        # Charts for numerical columns
        if data[column].dtype in ['float64', 'int64']:
            plt.figure(figsize=(12, 4))

            # Histogram
            plt.subplot(1, 5, 1)
            plt.hist(data[column])
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Freq')
            plots.append(plt)

            # Scatter Plot
            plt.subplot(1, 5, 2)
            sns.scatterplot(data=data, x=column, y=data.columns[-1])
            plt.xlabel(column)
            plt.ylabel("Values")
            plt.title("Scatter Plot")
            plots.append(plt)

            # Box Plot
            plt.subplot(1, 5, 3)
            sns.boxplot(data=data, y=column)
            plt.title(f'Box Plot of {column}')
            plt.ylabel(column)
            plots.append(plt)

        # Bar Plot for categorical columns
        if data[column].dtype in ['object']:
            plt.subplot(1, 5, 5)
            value_counts = data[column].value_counts()
            plt.bar(value_counts.index, value_counts.values)
            plt.xlabel(column)
            plt.ylabel('Values')
            plt.title(f'Bar Plot of {column}')
            plots.append(plt)

    return plots
def remove_outliers(data):
    
    for col in data.select_dtypes(include=["int64", "float64"]):
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
    return data

def clean_data(data):
    
    # drop null Values 
    df =handling_missing_data(data)
    
    # Encode cateogrical data
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
    
    # Encode numerical data
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Remove_outliers
    df=remove_outliers(df)
    
    return df
def main():
    st.title(" Simple pycaret App")

    # Upload data
    st.sidebar.header("Upload your data and preprocess it")
    data_upload = st.file_uploader("Upload Dataset (CSV only)", type=["csv"])
    
    try:
        if data_upload:
            df = pd.read_csv(data_upload)
            st.dataframe(df)
            # Load data from the uploaded file
            data =clean_data(df)
            st.sidebar.success("Data loaded and preprocessed successfully!")
            st.sidebar.subheader("4.Data Visualization")
            if st.sidebar.checkbox("Data Visualization"):
                plots=visualize_data(data)
                for plot in plots:
                    st.pyplot(plot)

           
            Train_model(data)
    except Exception as e:
            st.error(f"Error: {str(e)}")   
if __name__ == "__main__":
    main()
# To run ---> streamlit run main.py