# CANpose

## Overview

Implementation of my work titled "Integrating Depth Cues: A Cross-Attention Framework for Human Pose Recognition", which is currently under review at a major venue. The proposed model leverages the combined benefits of cross-attention and convolution to achieve human pose recognition from depth-maps. An overview of the proposed **C**ross-**A**tention convolution **N**etwork for human **pose** recognition (**CANpose**) is given below. 

![An overview of the proposed model](/block_diagram.png)

## Usage

```python
model = canpose((width,height),number_classes).to(DEVICE)
#(width,height) is the dimesnions of the input images
```


## Credits

The code is inspired from [CoAtNet](https://github.com/chinhsuanwu/coatnet-pytorch)
