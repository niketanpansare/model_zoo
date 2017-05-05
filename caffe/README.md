<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by the Berkeley Vision and Learning Center (BVLC) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley. Caffe is released under the BSD 2-Clause license. (Source: http://caffe.berkeleyvision.org/)

# Creating Caffe models

  1. Search through this folder to see if there is a network that fits your need.
  2. See http://caffe.berkeleyvision.org/tutorial/ for creating custom network file.
  3. An online GUI tool used to visualize prototxt and generate prototxt for caffe: http://yanglei.me/gen_proto/ (Source code: https://github.com/yl-1993/GenProto/tree/structured)

# Converting lmdb files to jpeg

```python
# mkdir ~/mnist_output
import conversion_utils, os
home_dir = os.path.expanduser('~')
conversion_utils.save_lmdb('../examples/mnist/mnist_train_lmdb', os.path.join(home_dir, 'mnist_output'), os.path.join(home_dir, 'mnist_label.txt'))
```

# Converting pretrained caffe model (.caffemodel) to the format supported by SystemML (for example: csv)

1. Create a directory `~/vgg_weights`.

2. Use Caffe's python API to save the weights and bias in `VGG_ILSVRC_19_layers.caffemodel` into  `~/vgg_weights`. The `VGG_ILSVRC_19_layers_deploy.prototxt` describes the Network.

```python
import conversion_utils, os
home_dir = os.path.expanduser('~')
vgg_pretrained_weight_dir = os.path.join(home_dir, 'vgg_weights')
conversion_utils.convert_caffemodel('/home/biuser/VGG_trained_models/VGG_ILSVRC_19_layers_deploy.prototxt', '/home/biuser/VGG_trained_models/VGG_ILSVRC_19_layers.caffemodel', vgg_pretrained_weight_dir)
```

3. Copy `labels.txt` containing the labels with its associated IDs. For ImageNet dataset, please use [labels.txt](https://github.com/niketanpansare/model_zoo/blob/master/caffe/vision/vgg/ilsvrc12/VGG_ILSVRC_19_pretrained_weights/labels.txt)

```bash
1,"tench, Tinca tinca"
2,"goldfish, Carassius auratus"
3,"great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias"
4,"tiger shark, Galeocerdo cuvieri"
5,"hammerhead, hammerhead shark"
6,"electric ray, crampfish, numbfish, torpedo"
7,"stingray"
...
```

4. Test the weights using Caffe2DML. Please use PySpark shell to execute the below code:

```python
from systemml.mllearn import Caffe2DML
from pyspark.sql import SQLContext
import numpy as np
import urllib, os, scipy.ndimage
from PIL import Image
import systemml as sml
# ImageNet specific parameters
img_shape = (3, 224, 224)
num_classes = 1000
# Downloads a jpg image, resizes it to 224 and return as numpy array in N X CHW format
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/MountainLion.jpg/312px-MountainLion.jpg'
outFile = 'test.jpg'
urllib.urlretrieve(url, outFile)
input_image = sml.convertImageToNumPyArr(Image.open(outFile), img_shape=img_shape)
# Download the VGG network
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/vgg/ilsvrc12/VGG_ILSVRC_19_layers_solver.proto', 'VGG_ILSVRC_19_layers_solver.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/vgg/ilsvrc12/VGG_ILSVRC_19_layers_network.proto', 'VGG_ILSVRC_19_layers_network.proto')
home_dir = os.path.expanduser('~')
vgg_pretrained_weight_dir = os.path.join(home_dir, 'vgg_weights')
# Load the pretrained model and predict the downloaded image
vgg = Caffe2DML(sqlCtx, solver='VGG_ILSVRC_19_layers_solver.proto', weights=vgg_pretrained_weight_dir, input_shape=img_shape)
vgg.predict(input_image)
```

# References

  1. https://github.com/yl-1993/GenProto/tree/structured
  2. List of layers in Caffe: http://caffe.berkeleyvision.org/tutorial/layers.html
  3. List of solvers in Caffe: http://caffe.berkeleyvision.org/tutorial/solver.html
