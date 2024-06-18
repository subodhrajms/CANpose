# CANpose

## Overview

Imploementation of my work titled "A Cross-Attention Convolution Network for Human Pose Recognition", which is currently under review.

![An overview of the proposed model](/block_diagram.png)

## Usage

```python
model = canpose((width,height),number_classes).to(DEVICE)
#(width,height) is the dimesnions of the input images
```


## Credits

The code is inspired from [CoAtNet](https://github.com/chinhsuanwu/coatnet-pytorch)
