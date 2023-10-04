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
import os 
os.system('wget https://www.dropbox.com/s/0zdxn7tcahrowddmew6l3/neuralhash_128x96_seed1.dat')
os.system('wget https://www.dropbox.com/s/12fou4r5f014xd780hs71/model.onnx')
# https://www.dropbox.com/scl/fi/0zdxn7tcahrowddmew6l3/neuralhash_128x96_seed1.dat?rlkey=l2zxcq6tdtdil2abg7vezmcwq&dl=0
# https://www.dropbox.com/scl/fi/12fou4r5f014xd780hs71/model.onnx?rlkey=dhxscfocphkvu1r2ejlfw8c1h&dl=0

torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2017/09/11/15/58/sunset-2739472_1280.jpg', 'sunset.jpg')
torch.hub.download_url_to_file('https://i.imgur.com/W8aXbd2.png', 'rotate.png')

torch.hub.download_url_to_file('https://user-images.githubusercontent.com/1328/129860794-e7eb0132-d929-4c9d-b92e-4e4faba9e849.png', 'dog.png')
torch.hub.download_url_to_file('https://user-images.githubusercontent.com/1328/129860810-f414259a-3253-43e3-9e8e-a0ef78372233.png', 'same.png')

torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2021/08/23/17/53/cat-6568422_1280.jpg', 'cat1.jpg')
torch.hub.download_url_to_file('https://i.imgur.com/fMoVhSz.png', 'cat2.png')

torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2021/08/10/09/41/lesser-sand-plover-6535531_1280.jpg', 'bird1.jpg')
torch.hub.download_url_to_file('https://i.imgur.com/jDgKAC7.png', 'bird2.png')






# Load ONNX model
session = onnxruntime.InferenceSession('model.onnx', providers=['AzureExecutionProvider', 'CPUExecutionProvider'])

# Load output hash matrix
seed1 = open('neuralhash_128x96_seed1.dat', 'rb').read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)
seed1 = seed1.reshape([96, 128])

pre_text = "<p style='display: flex; flex-grow: 1; align-items: center; justify-content: center; padding: 2rem 1rem; font-size: 1.5rem; line-height: 2rem; font-weight: 400;'>{}</p>"

# Preprocess image
def inference(img, img2):
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
  
  image2 = Image.open(img2.name).convert('RGB')
  image2 = image2.resize([360, 360])
  arr2 = np.array(image2).astype(np.float32) / 255.0
  arr2 = arr2 * 2.0 - 1.0
  arr2 = arr2.transpose(2, 0, 1).reshape([1, 3, 360, 360])
  
  # Run model
  inputs2 = {session.get_inputs()[0].name: arr2}
  outs2 = session.run(None, inputs2)
  
  # Convert model output to hex hash
  hash_output2 = seed1.dot(outs2[0].flatten())
  hash_bits2 = ''.join(['1' if it >= 0 else '0' for it in hash_output2])
  hash_hex2 = '{:0{}x}'.format(int(hash_bits2, 2), len(hash_bits2) // 4)
  
  if hash_hex == hash_hex2:
    return pre_text.format("Same Hash"), pre_text.format(hash_hex), pre_text.format(hash_hex2)
  return pre_text.format("Different Hash"), pre_text.format(hash_hex), pre_text.format(hash_hex2)
   
title = "AppleNeuralHash"
description = "Gradio demo for Apple NeuralHash, a perceptual hashing method for images based on neural networks. It can tolerate image resize and compression. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://www.apple.com/child-safety/pdf/CSAM_Detection_Technical_Summary.pdf'>CSAM Detection Technical Summary</a> | <a href='https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX'>Github Repo</a> | <a href='https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX/issues/1'>Working Collision example images from github issue</a></p> "
examples = [['sunset.jpg','rotate.png'],['dog.png','same.png'],['cat1.jpg','cat2.png'],['bird1.jpg','bird2.png']]

gr.Interface(
    inference, 
    [gr.inputs.Image(type="file", label="Input Image"),gr.inputs.Image(type="file", label="Input Image")], 
    [gr.outputs.HTML(label="Comparison.."), gr.outputs.HTML(label="First Hash"), gr.outputs.HTML(label="Second Hash")],
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging=False
    ).launch(debug=True)