from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Define the input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the HTTP request
    data = request.json

    # Preprocess the input data
    input_data = np.array(data['input_data'])
    input_data = np.expand_dims(input_data, axis=0)

    # Set the input tensor data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the model
    interpreter.invoke()

    # Get the output tensor data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Convert the output data to a JSON response
    response = {'output_data': output_data.tolist()}
    return response

if __name__ == '__main__':
    app.run()
