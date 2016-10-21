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

# Dataset

The MNIST dataset was constructed from two datasets of the US National Institute of Standards and Technology (NIST). The training set consists of handwritten digits from 250 different people, 50 percent high school students, and 50 percent employees from the Census Bureau. Note that the test set contains handwritten digits from different people following the same split.

In the below example, we are using `mlextend` package to load the mnist dataset into Python NumPy arrays, but you are free to download it directly from http://yann.lecun.com/exdb/mnist/.

```
pip install mlextend
```

# Networks

## LeNet

Lenet is a simple convolutional neural network, proposed by Yann LeCun in 1998. It has 2 convolutions/pooling and fully connected layer. Similar to Caffe, the network has been modified to add dropout. For more detail, please see http://yann.lecun.com/exdb/lenet/

# Example

```python
# Download the MNIST dataset
from mlxtend.data import mnist_data
import numpy as np
from sklearn.utils import shuffle
X, y = mnist_data()
X, y = shuffle(X, y)
numClasses = np.unique(y).shape[0]
imgShape = (1, 28, 28)

# Download the Lenet network
import urllib
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/lenet/mnist/lenet.proto', 'lenet.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/lenet/mnist/lenet_solver.proto', 'lenet_solver.proto')

# Train Lenet On MNIST
from systemml.mllearn import Barista
from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
barista = Barista(sqlCtx, numClasses, 'lenet_solver.proto', 'lenet.proto', imgShape)
barista.fit(X, y)
```

# References

  1. Y. LeCun and C. Cortes. Mnist handwritten digit database. AT&T Labs [Online]. Available: http://yann. lecun. com/exdb/mnist, 2010.
  2. http://yann.lecun.com/exdb/mnist/
  3. https://github.com/BVLC/caffe/tree/master/examples/mnist
