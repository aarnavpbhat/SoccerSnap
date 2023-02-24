from django.shortcuts import render

import tensorflow as tf


def predict(request):
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get the input data from the request
    input_data = request.GET.get('input_data')

    # Convert the input data to a numpy array
    input_data = np.array(input_data.split(','), dtype=np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the model
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Convert the output tensor to a JSON string and return it
    return JsonResponse({'output_data': json.dumps(output_data.tolist())})
