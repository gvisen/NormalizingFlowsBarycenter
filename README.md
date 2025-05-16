# Computing Optimal Transport Maps and Wasserstein Barycenters Using Conditional Normalizing Flows

This repository contains the code for the ICML 2025 paper **Computing Optimal Transport Maps and Wasserstein Barycenters Using Conditional Normalizing Flows** by Gabriele Visentin and Patrick Cheridito.

Here you can find the code for:
- the high-dimensional location-scatter experiments (Section 5.2.2),
- the MNIST experiment (Section 5.2.3).

## Requirements

This implementation is written in Python, is both GPU and CPU-compatible, and builds on `PyTorch` and the `normflows` package. All experiments in the paper were run either on an NVIDIA GeForce RTX 4090 GPU with 24 GB of memory or an NVIDIA RTX 6000 Ada with 48 GB. 

[!CAUTION]
The code was tested exclusively on the package versions listed in `requirements.txt`.

[!IMPORTANT]
To run the MNIST dataset, download and save the MNIST dataset in the folder `data/`.

## Code overview

- `src/`: source code
    - `bases`: base distribution classes for Normalizing Flows
    - `datasets`: dataset class for conditional Normalizing Flows
    - `flows`: custom flows for conditional Normalizing Flows, they all subclass `normflows.flows.base.Flow`
    - `metrics`: model evaluation metrics (L2-UVP and BW-UVP)
    - `models`: model class for multi-scale conditional Normalizing Flow with custom methods for barycenter transport and sampling
    - `samplers`: samplers for location-scatter benchmark
    - `utils`: miscellenia, custom conditional neural networks, GPU memory tools, etc
- `data/`: folder for MNIST dataset (optional)

## Citation

If you use this code, please cite the corresponding paper.

## Credits

This code depends on the `normflows` [package](https://pypi.org/project/normflows/) for the implementation of Normalizing Flows. The code in `src.samplers` was initially inspired by (but does not depend on) [this GitHub repo](https://github.com/iamalexkorotin/Wasserstein2Barycenters) of the ICLR 2021 paper **Continuous Wasserstein-2 Barycenter Estimation without Minimax Optimization** by Alexander Korotin, Lingxiao Li, Justin Solomon and Evgeny Burnaev.



