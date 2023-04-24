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
    #gender = st.selectbox("Gender",("Male","Female"), label_visibility=st.session_state.visibility, disabled=st.session_state.disabled,)
    age = st.slider("Age", 2.0, 100, 10)
    ht = st.slider("Height(m)", 2.0, 2, 1)
    
with col2:
    #st.text("Pepal characteristics")
    petal_l = st.slider("Petal lenght (cm)", 1.0, 7.0, 0.5)
    petal_w = st.slider("Petal width (cm)", 0.1, 2.5, 0.5)
st.button("Predict type of Iris")
