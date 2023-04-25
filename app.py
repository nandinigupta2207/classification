import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
# -----------------------------------------------------------

df=pd.read_csv(r"https://raw.githubusercontent.com/nandinigupta2207/classification/main/ObesityDataSet_raw_and_data_sinthetic.csv")
df_prep = df.copy()

# create dummy variables
df_prep = pd.get_dummies(df_prep,columns=["Gender","family_history_with_overweight","FAVC","CAEC","SMOKE","SCC","CALC","MTRANS"])

# split dataset in features and target variable
# Features
X = df_prep.drop("NObeyesdad", axis = 1)

# Target variable
y = df_prep['NObeyesdad']

# import sklearn packages for data treatments
# Import train_test_split function

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

mm = MinMaxScaler()
X_train_mm_scaled = mm.fit_transform(X_train)
X_test_mm_scaled = mm.transform(X_test)
model=DecisionTreeClassifier()
clf_mm_scaled = model.fit(X_train_mm_scaled, y_train)
clf_scaled = model.fit(X_train_mm_scaled,y_train)
y_pred_mm_scaled = clf_scaled.predict(X_test_mm_scaled)
# -----------------------------------------------------------

st.title("Uncovering Hidden Relationships: Obesity, Lifestyle Expressions")
st.markdown("FIND YOUR WAY TO HEALTH")
st.header("LIFESTYLE CHOICES")
col1, col2 = st.columns(2)
with col1:
    # st.text("Sepal characteristics")
    gen = st.selectbox("Select your gender", options=["Male", "Female"])
    age = st.slider("Age", 100, 10)
    height = st.slider("Select your height", 1.0, 2.0, step=0.01, format="%0.2f")
    weight = st.slider("Select  your height", 0.0, 300.0)
    fm = st.selectbox("Family history of obesity", options=["Yes", "No"])
    favc = st.selectbox("Frequent consumption of high caloric food ", options=["Yes", "No"])
    fcvc = st.slider("Frequency of consumption of vegetables", 1.0, 4.0, step=0.1, format="%0.2f")
    ncp = st.slider("Number of main meals", 1.0, 5.0, step=0.1)

with col2:
    # st.text("Pepal characteristics")
    caec = st.selectbox("Consumption of food between meals", options=["Sometimes", "Frequently", "Always", "no"])
    smoke = st.selectbox("do you smoke", options=["yes", "no"])
    ch20 = st.slider("Consumption of water daily(L)", 1.0, 4.0, step=0.1)
    scc = st.selectbox("Calories consumption monitoring", options=["no", "yes"])
    faf = st.slider("Physical activity frequency per day", 1.0, 3.0, step=0.5)
    tue = st.slider("Time using technology devices", 0.0, 12.0)
    calc = st.selectbox("Consumption of alcohol", options=['no', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.selectbox("Mode of transportation",
                          options=['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

if st.button("Predict type of obesity"):
    inp = [age, height, weight, fcvc, ncp, ch20, faf, tue]
    if gen == 1:
        inp.append(1)
        inp.append(0)
    elif gen == 0:
        inp.append(0)
        inp.append(1)
    fm = int(input("Family history with obesity: yes(1), no(0)"))
    if fm == 0:
        inp.append(1)
        inp.append(0)
    elif fm == 1:
        inp.append(0)
        inp.append(1)
    favc = int(input("Frequent consumption of high caloric food: yes(1), no(0)"))
    if favc == 0:
        inp.append(1)
        inp.append(0)
    elif favc == 1:
        inp.append(0)
        inp.append(1)
    caec = int(input("Consumption of food between meals : Always(1),Frequently(2),Sometimes(3),No(4)"))
    if caec == 1:
        inp.append(1)
        inp.append(0)
        inp.append(0)
        inp.append(0)
    elif caec == 2:
        inp.append(0)
        inp.append(1)
        inp.append(0)
        inp.append(0)
    elif caec == 3:
        inp.append(0)
        inp.append(0)
        inp.append(1)
        inp.append(0)
    else:
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(1)
    smoke = int(input("Do you smoke: yes(1), no(0)"))
    if smoke == 0:
        inp.append(1)
        inp.append(0)
    elif smoke == 1:
        inp.append(0)
        inp.append(1)
    scc = int(input("Do you monitor your calorie consumption: yes(1), no(0)"))
    if scc == 0:
        inp.append(1)
        inp.append(0)
    elif scc == 1:
        inp.append(0)
        inp.append(1)
    calc = int(input("Consumption of alcohol: Always(1),Frequently(2),Sometimes(3),No(4)"))
    if caec == 1:
        inp.append(1)
        inp.append(0)
        inp.append(0)
        inp.append(0)
    elif caec == 2:
        inp.append(0)
        inp.append(1)
        inp.append(0)
        inp.append(0)
    elif caec == 3:
        inp.append(0)
        inp.append(0)
        inp.append(1)
        inp.append(0)
    else:
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(1)
    mtrans = int(input(
        "What mode of transportation do you use: Automobile(1), Bike(2), Motorbike(3), Public Transport(4), Walking(5)"))
    if mtrans == 1:
        inp.append(1)
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(0)
    elif mtrans == 2:
        inp.append(0)
        inp.append(1)
        inp.append(0)
        inp.append(0)
        inp.append(0)
    elif mtrans == 3:
        inp.append(0)
        inp.append(0)
        inp.append(1)
        inp.append(0)
        inp.append(0)
    elif mtrans == 4:
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(1)
        inp.append(0)
    else:
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(1)
    
    st.write(input)

    input_arr = np.array(input)
    input_arr_scaled = mm.transform(input_arr)

    # make prediction
    result = model.predict(input_arr_scaled)[0]
    st.success(f'The obesity type i{result}')
