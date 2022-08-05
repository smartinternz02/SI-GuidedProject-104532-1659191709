import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask,request,render_template,redirect,url_for 
import os 
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)

#load both the vegetables and fruit models
#model = load_model("fruit.h5")
model1=load_model("vegetable.h5")  

#home page
@app.route('/')
def home():
    return render_template("home.html")

#prediction page
@app.route('/prediction')
def prediction():
    return render_template("predict.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        # Get the file name from the post request
        f=request.files['image']
        # Save the file to ./uploads
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(filepath)
        img=image.load_img(filepath,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['plant']
        print(plant)
        if(plant=="vegetable"):
            preds=model1.predict(x)
            print(preds)
            preds=np.argmax(model1.predict(x),axis=1)
            index=['Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy',
      'Tomato___Bacterial_spot','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot']
            index[preds[0]]
            df=pd.read_excel('precautions - veg.xlsx')
            print(df.iloc[preds[0]]['caution'])
            
        else:
            preds=model1.predict(x)
            df=pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds[0]]['caution'])
        return df.iloc[preds[0]]['caution']
          
if __name__=='__main__':
    app.run(debug=False)