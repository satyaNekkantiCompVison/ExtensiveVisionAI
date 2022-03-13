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

<img width="529" alt="EncodedImage" src="https://user-images.githubusercontent.com/90888045/158039468-c142ff2a-18c3-4edd-a15d-f832d26ffe45.png">


## How  dxN Box embeddings are created ?
- These are outputs for object queires for things and stuff like cow and background classes and output of encoded image passed to transformer decoder 

<img width="540" alt="Box_embeddings" src="https://user-images.githubusercontent.com/90888045/158039472-2db7b44e-0b37-4c78-a6b0-2fb9dd1e3d3b.png">


## Generation NxMxH/32xW/32 maps
- When we send encoded image to Multi Head attention Layer, for each of the Box embedding we get an attention layer map.

<img width="208" alt="Outptut_of_attention_layer" src="https://user-images.githubusercontent.com/90888045/158039476-250c9ccf-8a62-4e28-8692-d9358676c489.png">


## concatenate these maps with Res5 Block
- The Attention maps output of Multi Head attention layer is pass through a FPN(Feature Pyramid Network) CNN stype Network which is created using the intermediate activation functions which we used eariler to create the enocded image along with we add few upscale layers.
- Intial input to FPN network is H/32 x W/32 the output if FPN netowrk is mask layers with H/4 x W/4 , here we upscale the image to 8 times.
- The final output we get is mask-layers of each individual bounding box

<img width="440" alt="FPN_mask" src="https://user-images.githubusercontent.com/90888045/158039482-44f489a0-ea69-4c5a-a17b-7c18ecc8d0e8.PNG">


## Filtering of mask layer
- Once we get the individual mask layer for each class i.e foreground and background, we pass the mask layer to input image finally  we get individual class with pixel wise annotation.

<img width="400" alt="pixel_wise" src="https://user-images.githubusercontent.com/90888045/158039483-f40ced7e-3533-4e78-98b5-c7146614e911.PNG">


 
