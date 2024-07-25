import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model._base import _preprocess_data

main=Flask(__name__)
app=main

xgb_model=pickle.load(open('models/churn_pred.pkl','rb'))

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Slevel=float(request.form.get('Satisfaction Level'))
        levaluation=float(request.form.get("Last Evaluation Score"))
        noprojects=int(request.form.get('Number Of Projects'))
        averagehoursmonthly=int(request.form.get("Average Monthly Hours Worked"))
        timespendtocompany=int(request.form.get("Time Spend To Company Years"))
        anyworkacc=int(request.form.get("Any Work Accident"))
        anypromotion=int(request.form.get("Any Promotion Last 5 Years"))
        salarylow=int(request.form.get("Salary Low"))
        salarymedium=int(request.form.get("Salary Medium"))

        new_data=[Slevel,levaluation,noprojects,averagehoursmonthly,timespendtocompany,anyworkacc,anypromotion,salarylow,salarymedium]
        result=xgb_model.predict([new_data])
        
        return render_template('home.html',results=result[0])

    else:
        return render_template("home.html")



@app.route("/")
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
