import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
import pycaret as py
from pycaret.classification import *
# Load data in various formats, including CSV, Excel
def load_data(file_path):
    
    # Check if data in excel file 
    if file_path.name.endswith('.csv'):
        df = pd.read_csv(file_path)
        
    elif file_path.name.endswith('xlsx')or file_path.name.endswith("xls"):
        df = pd.read_excel(file_path)
        
    elif file_path.name.endswith('sql'):
        df = pd.read_sql(file_path)
        
    # Avoid error in file format
    else:
        raise ValueError("unsupported file")
    
    
    return df

def handling_missing_data(df):
    
   
    st.subheader('Drop Columns')
    columns_to_drop = st.multiselect("Select columns that you want to drop", options=df.columns)
    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)
    cat_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
      
    if len(cat_cols) > 0:
        cat_missing = st.selectbox("How do you want to handle missing values in categorical columns?", ["most frequent", "Don't handle (it will replace by No value instead of np.nan)"])
        if cat_missing == "most frequent":
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            for col in cat_cols:
                df[col].fillna("No Value", inplace=True)

    if len(numerical_cols) > 0:
        num_missing = st.selectbox("How do you want to handle missing values in numerical columns?", ["mean", "median", "mode"])
        if num_missing == "mean":
            for col in numerical_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        elif num_missing == "median":
            for col in numerical_cols:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            for col in numerical_cols:
                df[col].fillna(0, inplace=True)

    
    return df

def Train_model(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if chosen_target in numerical_cols :
        problem_type="Regression"
    elif chosen_target in cat_cols or df[chosen_target].dtype.name == 'object' :
        problem_type="Classification"    

     
    if st.button('Run Modelling'):
        if problem_type == 'Regression':
            
            for column in df.columns:
                # Encode numerical data
                scaler = StandardScaler()
                numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            from pycaret.regression import setup, compare_models,pull, save_model
            
            #Train model
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
            # Encode cateogrical data
            le = LabelEncoder()
            
            df[chosen_target] = le.fit_transform(df[chosen_target])
            #else :
                #df = pd.get_dummies(df, columns=[chosen_target])
            #Train_model    
            setup(df, target=chosen_target, verbose=False)
            st.success('Setup Complete')
            st.subheader('Model Comparison')
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
        
        #st.write("best model is ",best_model)
        #save_model(best_model, 'best_model')
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
    
    
    for column in df.columns:
        if df[column].dtype == 'bool':
            df[column] = df[column].astype(int)
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
    data_upload = st.file_uploader("Upload Dataset ")
    
    try:
        if data_upload:
            df = load_data(data_upload)
            st.dataframe(df)
            st.write('The shape of data : ',df.shape)    
            st.subheader('Data Types Of Columns')
            st.write(df.dtypes) 
            st.write("null values:", df.isna().sum())

            # Load data from the uploaded file
            data =clean_data(df)
            st.sidebar.success("Data loaded and preprocessed successfully!")
            
           
            
            Train_model(data)
            st.sidebar.subheader("4.Data Visualization")
            if st.sidebar.checkbox("Data Visualization"):
                plots=visualize_data(data)
                for plot in plots:
                    st.pyplot(plot)

    except Exception as e:
            st.error(f"Error: {str(e)}")   
if __name__ == "__main__":
    main()
# To run ---> streamlit run main.py
