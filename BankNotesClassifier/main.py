import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle


st.set_page_config(page_title='Bank Note Detection')
#make a title for your webapp
st.title("Bank Note Detection")
Logistic_regression,Decision_trees,Random_Forests = st.beta_columns(3)
with Logistic_regression:
    st.button('Logistic_regression')
with Decision_trees:
    st.button('Decision_trees')
with Random_Forests:
    st.button('Random_Forests')
KNN,SVC = st.beta_columns(2)    
with KNN:
    st.button('K-Nearest Neighbours')
with SVC:
    st.button('Support Vector Classifier') 
        
#lets try a both a text input and area as well as a date
field_1 = st.number_input('Variance') 
field_2 = st.number_input('Skewness')
field_3 = st.number_input('Curtosis')
field_4 = st.number_input('Entropy')

button = st.button('Submit')
if button:
    if Logistic_regression:
        infile = open('logistic_reg.pkl','rb')
        model = pickle.load(infile)
        pred = model.predict([[field_1,field_2,field_3,field_4]])
        infile.close()
    elif Decision_trees:
        infile = open('decision_tree.pkl','rb')
        model = pickle.load(infile)
        pred = model.predict([[field_1,field_2,field_3,field_4]])
        infile.close()
    elif Random_Forests:
        infile = open('random_forest.pkl','rb')
        model = pickle.load(infile)
        pred = model.predict([[field_1,field_2,field_3,field_4]])
        infile.close()
    elif KNN:
        infile = open('knn.pkl','rb')
        model = pickle.load(infile)
        pred = model.predict([[field_1,field_2,field_3,field_4]])
        infile.close()
    else:
        infile = open('svc_rbf.pkl','rb')
        model = pickle.load(infile)
        pred = model.predict([[field_1,field_2,field_3,field_4]])
        infile.close()


    if pred[0]==0:
        st.text("The Note is Real")
    else:
        st.text("The Note is Fake")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.write("Developed by Deep Gojariya.")