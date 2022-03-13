#  DETR for Panoptic Segmentation
Workflow for panoptic segmentation
- Train DETR to predict boxes for things(foreground) and stuff(background) objects
- Freeze the weights and train mask head for 25 epochs

## Architecture for Panoptic Segmentation
<img width="894" alt="panoptic_architecture_detr" src="https://user-images.githubusercontent.com/90888045/157561189-055cfa9e-ad0c-4816-9741-08372962c091.PNG">


## Creation of Encoded Image
- Step1:
	We send the image through the Backbone ResNet50 Conovolution Neural Network, also we save the activations from intermediate functions which are used in future for creating mask
- Step2:
	The output of CNN is passed through the transformer encoder

## How  dxN Box embeddings are created ?
- These are outputs for object queires for things and stuff like cow and background classes and output of encoded image passed to transformer decoder 

## Generation NxMxH/32xW/32 maps
- When we send encoded image to Multi Head attention Layer, for each of the Box embedding we get an attention layer map.

## concatenate these maps with Res5 Block
- The Attention maps output of Multi Head attention layer is pass through a FPN(Feature Pyramid Network) CNN stype Network which is created using the intermediate activation functions which we used eariler to create the enocded image along with we add few upscale layers.
- Intial input to FPN network is H/32 x W/32 the output if FPN netowrk is mask layers with H/4 x W/4 , here we upscale the image to 8 times.
- The final output we get is mask-layers of each individual bounding box

## Filtering of mask layer
- Once we get the individual mask layer for each class i.e foreground and background, we pass the mask layer to input image finally  we get individual class with pixel wise annotation.

 
