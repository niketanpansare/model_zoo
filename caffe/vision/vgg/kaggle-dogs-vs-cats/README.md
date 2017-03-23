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

```python
TODO:
```
