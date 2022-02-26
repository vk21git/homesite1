from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
import os
from os.path import join, dirname, realpath
import pickle
import pandas as pd
#from sklearn.externals import joblib
import joblib


app = Flask(__name__)
#model = pickle.load(open('SG_finalized_model.pkl', 'rb'))

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
#UPLOAD_FOLDER = 'static/files'
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')


# Get the uploaded files
# @app.route("/", methods=['post'])
def uploadfiles():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '':
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          # set the file path
           uploaded_file.save(file_path)
          # save the file
      return redirect(url_for('index'))

#@app.route("/", methods=['post'])
@app.route('/',methods=['POST'])
def predict():
    #a = uploadfiles()
    # get the uploaded file
    # uploaded_file = request.files['file']
    # if uploaded_file.filename != '':
    #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    #     # set the file path
    #     uploaded_file.save(file_path)
    # #return redirect(url_for('index'))
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

    os.chdir("static/files")
    filename = 'vikas_model.pkl'
    RFE_columns = pd.read_csv('RFE_features_1.csv').columns
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    #
    # Load the test set
    df_test = pd.read_csv('sample.csv')

    ## just take the required columns requied for predicting on it

    RFE_columns = [col for col in RFE_columns if col not in 'QuoteConversion_Flag']  # Test wont have the label column
    df_test = df_test[RFE_columns]
    df_test.drop(columns=['Original_Quote_Date', 'SalesField8'], axis=1, inplace=True)

    # Predict on it using the GBM2 pipeline . Here pipe line has made a task easy as we do not have to store features
    #    y_test = GBM2.predict_proba(df_test)
    y_test = loaded_model.predict_proba(df_test)

    print(y_test)
    print(y_test[:, 1])

    output = round(y_test[:, 1][0], 0)
    print(output)

    #clf = joblib.load('SG_finalized_model.pkl')

    if output>0.5:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return jsonify({'prediction': prediction})
    return redirect(url_for('index'))

if (__name__ == "__main__"):
     app.run(port = 5000,debug=True)