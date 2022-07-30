# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:15:28 2021

@author: deepak
"""
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("decision_model.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(UserID, Gender,Age,EstimatedSalary):
  output= model.predict(sc.transform([[Age,EstimatedSalary]]))
  print("Purchased", output)
  if output==[1]:
    prediction="Item will be purchased"
  else:
    prediction="Item will not be purchased"
  print(prediction)
  return prediction
def main():
    st.title("Item Purchase Prediction")
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Deep Lab Experiment Deployment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    UserID = st.text_input("UserID","Type Here")
    Gender = st.text_input("Gender","Type Here")
    Age = st.text_input("Age","Type Here")
    EstimatedSalary = st.text_input("EstimatedSalary","Type Here")
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(UserID, Gender,Age,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.text("Developed by Deepak Moud")
      st.text("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()
   
