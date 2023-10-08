#!/usr/bin/env python3 
import numpy as np
import sys

with open("labels.txt", "r") as f:
   labels=f.readlines()

from tflite_runtime.interpreter import Interpreter
from PIL import Image


# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path="densenet_1_default_1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('Input Details',input_details[0]['shape'])


for image in sys.argv[1:]:
  im = Image.open(sys.argv[1])
  im.load()
  b = im.resize((224,224))
  
  input_data = np.array(b, dtype=np.float32).reshape(input_details[0]['shape'])
  # These are based on sample code- it worked for my Westie but missed more than it hit. 

  input_data = (input_data-127.5)/127.5

  # Used an embedded python which didn't have PIL JPEG installed. 
#  np.save(image+".npy", input_data)
#  input_data2 = np.load(image+".npy")

  input_data2 = input_data
  interpreter.set_tensor(input_details[0]['index'], input_data2)
  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  
  # Figure out what we think this is. 
  line = 1
  for i in output_data[0]:
    if i > 0.3:
      print(line,i, labels[line-1]) 
    line+=1 
