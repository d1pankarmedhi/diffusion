<div align="center">
  <h1>Diffusion Model</h1>
  <p>A PyTorch implementation of a Diffusion Model for image generation.</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white)

</div>

## Overview

The project implements a diffusion model that gradually converts random noise into meaningful images through an iterative denoising process. The complete diffusion process can be divided into two phases: **Forward and Reverse Diffusion**.

### Forward Diffusion (Adding Noise to image)

This process involves adding random noise to the input image, progressively corrupting them in a step by step process. This processes is also referred to as Markov Chain, where the state at each step depends on the previous step.

The noise addion continues till the input image becomes pure noise, represented by Gaussian distribution.

### Reverse Diffusion (Predicting & removing noise)

In this process, the model learns to undo the forwared diffusion process by learning to remove the noise step by step, to recontruct the original data.

The model start with pure noise and learn to transform it into a coherent image. Here, a neural network, such as UNet (or Transformer) learns to predict the noise added at each step in the forward process. It learns to predict what noise to remove at each step.

Iteratively, the models learns to remove the predicted noise from the image at each time step, gradually refining the input into a fine output image.

## License

See [LICENSE](LICENSE) file for details.
