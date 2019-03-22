## **A Deep Path Planning Algorithm Based on CNNs for Perception Images**

### Introduction

Path planning for robots navigation, commercial computer games, off-line map applications and many other fields is an ongoing research. There have raised several methods derived from the traditional A-star algorithm due to its efficiency in the past few years. As for the limitations of algorithms in global path
planning, we introduce a novel method based on deep learning. We present a novel path planning algorithm combined with convolutional neural networks (CNNs) to learn a target oriented end-to-end model from the input of images. The deep neural network proved to be efficient and effective in feature
extracting in our experiments too. The model can transfer the scene understanding and navigation knowledge gained from one environment to another unseen ones. Finally, this method can not only maintain the optimality of the path, but can also greatly accelerate the computation.

### Experimental method

We use the modified A-star algorithm to produce a dataset of 27000 images. The size of each image is 400 × 400. And the data set consists of 7000 images with fixed initial positions and target positions, while the other 20000 images have their initial positions changed within a certain range. All of these images have different obstacles and threatened areas from size and coordinate position.

##### Construct a modified A-star algorithm to create paths for global maps.

![all-way.png](https://i.loli.net/2019/03/22/5c944d5cb7824.png)

<center> Fig.1 The path planned by a modified 24-way search A-star Algorithm </center>

##### Construct an end-to-end model upon the convolutional neural networks.

| Layer | Structure                                                    |
| ----- | ------------------------------------------------------------ |
| 1     | composite convolutional layers (32 filters of 7×7  and   1 max-pooling layer) |
| 2     | composite convolutional layers (64 filters of 5×5  and   1 max-pooling layer) |
| 3     | composite convolutional layers (128\256\128 filters   of 3×3  and 1 max-pooling layer) |
| 4     | composite convolutional layers (512 filters of 3×3  and   1 max-pooling layer) |
| 5/6/7 | Fully-connected  layers: 1024, 1024 and 50 units respectively. |

![convolutional_layers.png](https://i.loli.net/2019/03/22/5c944fee7c99f.png)

<center> Fig.2 Illustration of the composite convolutional layer for layer2 </center>

### Results and discussion

##### 1 Trajectory Comparison (Fig.3)

![comp1.png](https://i.loli.net/2019/03/22/5c945066351ec.png)
![comp2.png](https://i.loli.net/2019/03/22/5c94506635e31.png)
![comp3.png](https://i.loli.net/2019/03/22/5c945066358da.png)

<center>Fig.3 Results generated by our deep path planning algorithm. The red line represents the path planned by the modified A-star algorithm, and the black path is predicted by our end-to-end model.</center>

##### 2 Conclusions

(a)Given perception images, our approach is able to compute the required paths for a mobile robot or a computer game agent. 

(b)It greatly reduces the runtime from the second level to the millisecond level while preserving the excellent features of the modified A-star algorithm. 