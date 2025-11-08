<div align="center">
  <h1>Diffusion Model</h1>
  <p>A PyTorch implementation of the Diffusion Model for image generation.</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white)

</div>

The project implements a diffusion model that gradually converts random noise into meaningful images through an iterative denoising process. The complete diffusion process can be divided into two phases: **Forward and Reverse Diffusion**.

  <table>
    <tbody>
      <tr>
        <td> <img width="500" alt="image" src="https://github.com/user-attachments/assets/64ee6045-766b-43cd-9d14-d64a6bbcc0fc" /></td>
        <td> <img width="450" alt="image" src="https://github.com/user-attachments/assets/cd5fa8d2-cd96-4182-b47e-b1fbb55a805a" /></td>
      </tr>
    </tbody>
  </table>

## 🕸️ Forward Diffusion (Adding Noise to image)

This process involves adding random noise to the input image $x_0$, progressively corrupting it in a step-by-step process, for $t = 1, ..., T$, arriving at $x_t$. This process is also referred to as a Markov Chain, where the state at each step depends on the previous step.

At each step $t$, we add a little bit of Gaussian noise. $\beta_t$ is a small value of noise added to the image, and 
> $\alpha_t = 1 - \beta_t$

We keep adding noise over time, defining a cumulative product

> $\overline{\alpha_t}$ = $\sum_{s=1}^{t} \alpha_s$

and over time, after $t$ steps, only a part of the image survives. 

Thus, the direct formula for the noisy image can be expressed as:

> $x_t = \sqrt{\overline{\alpha}} x_0 + \sqrt{1 - \overline{\alpha_t}}ϵ$

where,

  - $x_t$ = noisy image at step t
  - $x_0$ = original image
  - $ϵ \sim N(0, I)$ = random Gaussian noise

In short,

- $\sqrt{\overline{\alpha}}x_0$ = how much clean image survives
- $\sqrt{1 - \overline{\alpha_t}}ϵ$ = how much noise is mixed in the image.

Also, referred to as the forward diffusion equation.

<div align="center">
<img width="700"  alt="forward-diffusion" src="https://github.com/user-attachments/assets/41499028-e0fe-4ccb-ba32-a8f7495335c9" />
<p>Fig: Forward Diffusion</p>
</div>


## 🦋 Reverse Diffusion (Predicting & removing noise)

In this process, the model learns to undo the forward diffusion process by learning to remove the noise step by step, to reconstruct the original data.

The model starts with pure noise $x_t$ (noisy image) and learns to transform it into a coherent image $x_0$. 

Here, a neural network, such as [UNet](https://arxiv.org/pdf/1505.04597) (Ronneberger et al. (2015)) (or Transformer), learns to predict the noise added at each step in the forward process. It learns to predict what noise to remove at each step.

<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/241c1393-6146-4926-a11c-e8abd2ce047b" />
<p><em>Fig: UNet Architecture</em></p>
</div>

Iteratively, the model learns to remove the predicted noise from the image at each time step, gradually refining the input into a fine output image.

<div align="center">
<img src="https://github.com/user-attachments/assets/79cb5a4b-0a80-4676-8a56-1f18110b9f0d" width="350" />
</div>


## License

See [LICENSE](LICENSE) file for details.
