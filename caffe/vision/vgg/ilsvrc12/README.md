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
  3. Start pyspark shell: `pyspark --master local[*] --driver-memory 5g  --driver-class-path SystemML.jar`

```python
from systemml.mllearn import Barista
from pyspark.sql import SQLContext
import numpy as np
import urllib, os, scipy.ndimage
from PIL import Image

# ImageNet specific parameters
img_shape = (3, 224, 224)
num_classes = 1000

# Utility method that downloads a jpg image, resizes it to 224 and return as numpy array in N X CHW format
def downloadAsNumPyArray(url):
    outFile = 'test.jpg'
    urllib.urlretrieve(url, outFile + '.tmp')
    Image.open(outFile + '.tmp').resize( (224,224), Image.LANCZOS).save(outFile)
    os.remove(outFile + '.tmp')
    t = np.einsum('ijk->kij', scipy.ndimage.imread(outFile))
    return t.reshape(1, t.size)


input_image = downloadAsNumPyArray('https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/MountainLion.jpg/312px-MountainLion.jpg')
assert input_image.shape == (1, img_shape[0]*img_shape[1]*img_shape[2])

vgg_dir = '< path to model_zoo/caffe/vision/vgg/ilsvrc12>'
sql_ctx = SQLContext(sc)
vgg = Barista(sql_ctx, num_classes, os.path.join(vgg_dir, 'VGG_ILSVRC_19_layers_solver.proto'), os.path.join(vgg_dir, 'VGG_ILSVRC_19_layers_network.proto'), img_shape)
vgg.load(os.path.join(vgg_dir, 'VGG_ILSVRC_19_pretrained_weights'))
vgg.predict(input_image)
```
