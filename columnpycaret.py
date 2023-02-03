
import streamlit as st
import pandas as pd
import numpy as np
import pickle  
import os
from operator import index
import pandas_profiling
from pandas_profiling import ProfileReport  
from streamlit_pandas_profiling import st_profile_report


with st.sidebar:
    st.image("https://miro.medium.com/max/1400/1*r-8_klG8IIlRTE5zJpg6sA.png")
    st.title("MachineLearningAppWeb")
    choice=st.radio("Navigation",["Upload","Profiling","ColumnModelPredict", "BuildModel", "ColumnModelCompare"])
    st.info('This application allows you to build an automated ML pipeline using Streamlit.')
    st.info('Example dataset1: https://raw.githubusercontent.com/yimeiwen/automatedMLapp/main/Home_n_Map.csv')
    st.info('Example dataset2: https://raw.githubusercontent.com/yimeiwen/InstrumentPerformanceEvaluation/main/waters_pass_fail_column.csv')
if os.path.exists('download.csv'):
    df=pd.read_csv('download.csv', index_col=None)



if choice=="Upload":
    st.title("Upload your data for modelling")
    file=st.file_uploader("Upload your dataset (csv file)")
    if file:
        df=pd.read_csv(file)
        df.to_csv("download.csv", index=None)
        st.dataframe(df)

#import pandas_profiling
#from pandas_profiling import ProfileReport  
if choice=="Profiling":
    st.title("Exploratory Data Analysis")
    report=df.profile_report()
    st_profile_report(report)


if choice=="ColumnModelPredict":
    Columninput=st.container()
    with Columninput:
        st.text("Enter the value of Column Tested")
        Area=st.number_input("Area Count")
        Area_P=st.number_input("%Area")
        Height=st.number_input("Height")
        Width=st.number_input("Width")
        USP_Resolution=st.number_input("USP Resolution")
        Asym_10=st.number_input(" Asym@10")
        Asym=st.number_input("Asym")
        USP_Tailing=st.number_input("USP_Tailing")
        USP_Plate_Count=st.number_input("USP_Plate_Count")
        Width_Baseline=st.number_input("Width@Baseline")
        Width_Tang=st.number_input("Width@Tangent")
        Width_5=st.number_input("Width@5")
        Width_50=st.number_input("Width@50")
        load_model = pickle.load(open('uplc_column_performance_score.pkl', 'rb'))
        X=[[Area,Area_P,Height,Width,USP_Resolution,Asym_10,Asym,USP_Tailing,USP_Plate_Count,Width_Baseline,Width_Tang,Width_5,Width_50]]
        load_model = pickle.load(open('uplc_column_performance_score.pkl', 'rb'))
        prediction=load_model.predict(X)
        prediction_proba=load_model.predict_proba(X)
    
        st.header("Predicted Standard")
        prediction
        prediction_proba
        def load_data():
        	return pd.DataFrame({"0":["15T"],"1":["20T"],"2":["25T"],"3":["30T"],"4":["35T"]})
        df1=load_data()
        st.text("Class Labels and its Corresponding Standard")
        st.dataframe(df1)



from pycaret.regression import setup, compare_models,pull, save_model
if choice=="BuildModel":
    st.title("Build your model")
    target=st.selectbox('Select Your Target',df.columns)
    setup(df,target=target)
    setup_df=pull()
    st.info("This is a Model Building Experiment")
    st.dataframe(setup_df)
    best_model=compare_models()
    compare_df=pull()
    st.info("This is the Model Parameters")
    st.dataframe(compare_df)
    best_model
    
   



if choice=="ColumnModelCompare":
    st.header("Random Forest Prediction")
    st.header("Logistics Regression Prediction")
    st.header("KNN Prediction")
  