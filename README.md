<div align="center">
  <h1>Diffusion Model</h1>
  <p>A PyTorch implementation of a Diffusion Model for image generation.</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white)

</div>

## Overview

The project implements a diffusion model that gradually converts random noise into meaningful images through an iterative denoising process. The complete diffusion process can be divided into two phases: **Forward and Reverse Diffusion**.

### Forward Diffusion (Adding Noise to image)

This process involves adding random noise to the input image, progressively corrupting them in a step by step process. This processes is also referred to as Markov Chain, where the state at each step depends on the previous step.

The noise addition continues till the input image becomes pure noise, represented by Gaussian distribution.

<img width="1182" height="184" alt="image" src="https://github.com/user-attachments/assets/41499028-e0fe-4ccb-ba32-a8f7495335c9" />


### Reverse Diffusion (Predicting & removing noise)

In this process, the model learns to undo the forward diffusion process by learning to remove the noise step by step, to reconstruct the original data.

The model starts with pure noise and learn to transform it into a coherent image. Here, a neural network, such as [UNet](https://arxiv.org/pdf/1505.04597) (Ronneberger et al. (2015)) (or Transformer) learns to predict the noise added at each step in the forward process. It learns to predict what noise to remove at each step.

<div align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/241c1393-6146-4926-a11c-e8abd2ce047b" />
<p><em>Fig: UNet Architecture</em></p>
</div>

Iteratively, the model learns to remove the predicted noise from the image at each time step, gradually refining the input into a fine output image.

## Training and Inference

The PyTorch implementation creates a very small model, trained on the FashionMNIST dataset for 5 epochs only. It is only done for educational purposes, keeping the hardware requirements in check while making sure the concepts are well defined.

<div align="center">
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2d2e8b67-0bd3-4d8f-9a36-de7ebe02b3e0" width="500" />
      <p><em>Fig: Sample Generations</em></p>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/79cb5a4b-0a80-4676-8a56-1f18110b9f0d" width="350" />
      <p><em>Fig: Samples per timestep t</em></p>
    </td>
  </tr>
</table>
</div>

## License

See [LICENSE](LICENSE) file for details.
