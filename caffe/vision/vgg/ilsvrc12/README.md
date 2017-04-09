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

# VGG

If you plan to use the network or the model, please cite "K. Simonyan, A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition, arXiv technical report, 2014". Also, please read https://arxiv.org/pdf/1409.1556.pdf to understand the parameters, architecture and training procedure.

## VGG 19-layer model

![VGG 19-layer network](VGG_ILSVRC_19_layers_network.png)

### Example

  1. Install packages used in the below example: `pip install Pillow`
  2. Download the trained model and network: `git clone https://github.com/niketanpansare/model_zoo.git`
  3. Start pyspark shell: `pyspark --driver-memory 5G --conf spark.driver.maxResultSize=0 --conf "spark.driver.extraJavaOptions=-Xmn500m -server" --driver-class-path SystemML.jar`

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

# Load the pretrained model and predict the downloaded image
home_dir = os.path.expanduser('~')
vgg_pretrained_weight_dir = os.path.join(home_dir, 'model_zoo', 'caffe', 'vision', 'vgg', 'ilsvrc12', 'VGG_ILSVRC_19_pretrained_weights')
vgg = Caffe2DML(sqlCtx, solver='VGG_ILSVRC_19_layers_solver.proto', weights=vgg_pretrained_weight_dir, input_shape=img_shape)
vgg.predict(input_image)
```
