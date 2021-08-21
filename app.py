# Copyright 2021 Asuhariet Ygvar
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

import sys
import onnxruntime
import numpy as np
from PIL import Image
import gradio as gr
import torch

torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2017/09/11/15/58/sunset-2739472_1280.jpg', 'sunset.jpg')

# Load ONNX model
session = onnxruntime.InferenceSession('model.onnx')

# Load output hash matrix
seed1 = open('neuralhash_128x96_seed1.dat', 'rb').read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)
seed1 = seed1.reshape([96, 128])

# Preprocess image
def inference(img):
  image = Image.open(img.name).convert('RGB')
  image = image.resize([360, 360])
  arr = np.array(image).astype(np.float32) / 255.0
  arr = arr * 2.0 - 1.0
  arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])
  
  # Run model
  inputs = {session.get_inputs()[0].name: arr}
  outs = session.run(None, inputs)
  
  # Convert model output to hex hash
  hash_output = seed1.dot(outs[0].flatten())
  hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
  hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
  
  return hash_hex
 
title = "AppleNeuralHash"
description = "Gradio demo for Apple NeuralHash, a perceptual hashing method for images based on neural networks. It can tolerate image resize and compression. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://www.apple.com/child-safety/pdf/CSAM_Detection_Technical_Summary.pdf'>CSAM Detection Technical Summary</a> | <a href='https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX'>Github Repo</a></p>"
examples = [['sunset.jpg']]

gr.Interface(
    inference, 
    gr.inputs.Image(type="file", label="Input"), 
    gr.outputs.Textbox(label="Output Text"),
    title=title,
    description=description,
    article=article,
    examples=examples
    ).launch(debug=True)