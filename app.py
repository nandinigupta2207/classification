import streamlit as st
import pandas as pd
import numpy as np
#from prediction import predict

st.title("Uncovering Hidden Relationships: Obesity, Lifestyle Expressions")
st.markdown("FIND YOUR WAY TO HEALTH")
st.header("LIFESTYLE CHOICES")
col1, col2 = st.columns(2)
with col1:
    #st.text("Sepal characteristics")
    gender = st.selectbox("Select your gender", options=["Male", "Female"])
    age = st.slider("Age",100, 10)
    height = st.slider("Select your height", 1.0, 2.0, step=0.01, format="%0.2f")
    weight=st.slider("Select  your height", 0.0, 300.0)
    history=st.selectbox("Family history of obesity",options=["Yes","No"])
    favc=st.selectbox("Frequent consumption of high caloric food ",options=["Yes","No"])
    fcvc=st.slider("Frequency of consumption of vegetables",1.0,4.0,step=0.01,format"%0.2f")
    ncp=st.slider("Number of main meals",1.0,5.0)
    
with col2:
    #st.text("Pepal characteristics")
    petal_l = st.slider("Petal lenght (cm)", 1.0, 7.0, 0.5)
    petal_w = st.slider("Petal width (cm)", 0.1, 2.5, 0.5)
st.button("Predict type of Iris")
