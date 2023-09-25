from flask import Flask,render_template, redirect, url_for, request 
import pickle
import numpy as np
#init app in flask
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))


#ameen added code here
print("helloooo")
#default rout
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    pred = model.predict(features)

    if pred == 1:
        strr = "you can eat the Pizza"
    else:
        strr = "you can't eat the Pizza"
    return render_template("index.html", text_prediction="you model output is:  {} and {}".format(pred,strr))

if __name__=='__main__':
    app.run()