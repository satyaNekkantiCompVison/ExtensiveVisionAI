#  DETR for Panoptic Segmentation
Workflow for panoptic segmentation
- Train DETR to predict boxes for things(foreground) and stuff(background) objects
- Freeze the weights and train mask head for 25 epochs

## Architecture for Panoptic Segmentation
<img width="894" alt="panoptic_architecture_detr" src="https://user-images.githubusercontent.com/90888045/157561189-055cfa9e-ad0c-4816-9741-08372962c091.PNG">
