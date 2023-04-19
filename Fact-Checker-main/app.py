from flask import Flask, render_template, jsonify, request
from main import main,main2
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # gets image file name from request
    imagefile = request.files['imagefile'].read()
    # converts file object to bytes
    imgbytes = np.fromstring(imagefile, np.uint8)
    print(imgbytes.shape)
    # converts bytes to image
    img = cv2.imdecode(imgbytes, cv2.IMREAD_COLOR)

    result = main(img)

    # outputs result in output_display div
    return jsonify({'result': result}) 

@app.route('/fact_check', methods=['POST'])
def fact_check():
    # Get the input text from the headline-input field
    input_text = request.form['headline-input']
    # Perform some processing (e.g. fact-checking) on the input text
    # and get the result
    result = main2(input_text)

    # Create a JSON response containing the result
    response = {'result': result}

    # Return the response as a JSON object
    return jsonify(response) 

if __name__ == '__main__':
    app.run(debug=False)