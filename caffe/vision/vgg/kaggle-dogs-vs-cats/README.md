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

This example demonstrates transfer learning using pre-trained VGG model and is based on [the Francis Chollet's tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

  1. Install packages used in the below example: `pip install Pillow`
  2. Download the trained model and network: `git clone https://github.com/niketanpansare/model_zoo.git`
  3. Start pyspark shell: `pyspark --master local[*] --driver-memory 20g  --driver-class-path SystemML.jar`
  4. Download train.zip from https://www.kaggle.com/c/dogs-vs-cats/data
  5. Freeze weight and bias of convolution layer by adding `param { lr_mult: 0 }`
  6. Modify `num_output` of last `InnerProduct` layers in the network proto from 4096, 4096, 1000 to 256, 256, 2 respectively. 

#### Preprocessing the dataset
```python
from systemml.mllearn import Caffe2DML
from pyspark.sql import SQLContext
import numpy as np
import urllib, os, scipy.ndimage
from pyspark.ml.linalg import Vectors
from pyspark import StorageLevel
import systemml as sml
from pyspark.sql.functions import rand 
# ImageNet specific parameters
img_shape = (3, 224, 224)
train_dir = '/home/biuser/dogs_vs_cats/train'
def getLabelFeatures(filename):
	from PIL import Image
	vec = Vectors.dense(sml.convertImageToNumPyArr(Image.open(os.path.join(train_dir, filename)), img_shape=img_shape)[0,:])
	if filename.lower().startswith('cat'):
		return (1, vec)
	elif filename.lower().startswith('dog'):
		return (2, vec)
	else:
		raise ValueError('Expected the filename to start with either cat or dog')

list_jpeg_files = os.listdir(train_dir)
# 10 files per partition
train_df = sc.parallelize(list_jpeg_files, int(len(list_jpeg_files)/10)).map(lambda filename : getLabelFeatures(filename)).toDF(['label', 'features']).orderBy(rand())
# Optional: but helps seperates conversion-related from training
train_df.write.parquet('kaggle-cats-dogs.parquet')
```

#### Fine-tuning

```python
from systemml.mllearn import Caffe2DML
from pyspark.sql import SQLContext
import numpy as np
import urllib, os, scipy.ndimage
import systemml as sml
# ImageNet specific parameters
img_shape = (3, 224, 224)
train_df = sqlCtx.read.parquet('kaggle-cats-dogs.parquet')

# Download the Proto files
import urllib
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/vgg/kaggle-dogs-vs-cats/VGG_ILSVRC_19_layers_network.proto', 'VGG_ILSVRC_19_layers_network.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/vgg/kaggle-dogs-vs-cats/VGG_ILSVRC_19_layers_solver.proto', 'VGG_ILSVRC_19_layers_solver.proto')

# Train
vgg = Caffe2DML(sqlCtx, solver='VGG_ILSVRC_19_layers_solver.proto', weights='/home/biuser/model_zoo/caffe/vision/vgg/ilsvrc12/VGG_ILSVRC_19_pretrained_weights', ignore_weights=['fc6', 'fc7', 'fc8']).set(input_shape=img_shape, debug=True, max_iter=500).setExplain(True)
vgg.fit(train_df)
vgg.save('kaggle-cats-dogs-model')
```
