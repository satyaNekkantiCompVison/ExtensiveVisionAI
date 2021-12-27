## Spatial Transformers

Spatial transformer networks boils down to three main components :

1. The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
2. The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.
3. The sampler uses the parameters of the transformation and applies it to the input image.