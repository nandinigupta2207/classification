import streamlit as st
import pandas as pd
import numpy as np
#from predictions import predict

st.title("Uncovering Hidden Relationships: Obesity, Lifestyle Expressions")
st.markdown("FIND YOUR WAY TO HEALTH")
st.header("LIFESTYLE CHOICES")
col1, col2 = st.columns(2)
with col1:
    #st.text("Sepal characteristics")
    g = st.selectbox("Select your gender", options=["Male", "Female"])
    age = st.slider("Age",100, 10)
    height = st.slider("Select your height", 1.0, 2.0, step=0.01, format="%0.2f")
    weight=st.slider("Select  your height", 0.0, 300.0)
    history=st.selectbox("Family history of obesity",options=["Yes","No"])
    favc=st.selectbox("Frequent consumption of high caloric food ",options=["Yes","No"])
    fcvc=st.slider("Frequency of consumption of vegetables",1.0,4.0,step=0.1,format="%0.2f")
    ncp=st.slider("Number of main meals",1.0,5.0,step=0.1)
    
with col2:
    #st.text("Pepal characteristics")
    caec = st.selectbox("Consumption of food between meals", options=["Sometimes", "Frequently","Always","no"])
    smoke=st.selectbox("do you smoke", options=["yes","no"])
    ch20=st.slider("Consumption of water daily(L)",1.0,4.0,step=0.1)
    scc=st.selectbox("Calories consumption monitoring", options=["no","yes"])
    faf=st.slider("Physical activity frequency per day",1.0,3.0,step=0.5)
    tue=st.slider("Time using technology devices",0.0,12.0)
    calc=st.selectbox("Consumption of alcohol", options=['no', 'Sometimes', 'Frequently', 'Always'])
    mtrans=st.selectbox("Mode of transportation", options=['Public_Transportation', 'Walking', 'Automobile', 'Motorbike','Bike'])
with open('rf_model.pkl', 'rb') as rf:
    model = pickle.load(rf)
# load the StandardScaler
with open('scaler.pkl', 'rb') as mm:
    scaler = pickle.load(mm)
def predict(Normal_Weight, Overweight_Level_I, Overweight_Level_II,Obesity_Type_I, Insufficient_Weight, Obesity_Type_II,Obesity_Type_III):
    
    # processing user input
    ocean = 0 if ocean_pro == '<1H OCEAN' else 1 if ocean_pro == 'INLAND' else 2 if ocean_pro == 'ISLAND' else 3 if ocean_pro == 'NEAR BAY' else 4
    med_income = median_income / 5
    lists = [Normal_Weight, Overweight_Level_I, Overweight_Level_II,Obesity_Type_I, Insufficient_Weight, Obesity_Type_II,Obesity_Type_III]
    
    df = pd.DataFrame(lists).transpose()
    # scaling the data
    scaler.transform(df)
    # making predictions using the train model
    prediction = model.predict(df)
    result = int(prediction)
    return result

   
st.button("Predict type of obesity")
if button:
        
        # make prediction
        result = predict(Normal_Weight, Overweight_Level_I, Overweight_Level_II,Obesity_Type_I, Insufficient_Weight, Obesity_Type_II,Obesity_Type_III)
        st.success(f'The obesity type is ${result}')
