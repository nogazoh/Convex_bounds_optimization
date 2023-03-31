# Multiple-Source-Adaptation-using-Variational-R-nyi-Bound-Optimization
This project contains code for our paper "Multiple-Source Adaptation using Variational Rényi Bound Optimization".

It contains pytorch implementation of Variational Inference model, using different loss functions:
  1. Maximizing the ELBO.
  2. Maximizing Rényi Lower Bound with positive alpha.
  3. Minimizing Rényi Upper Bound with negative alpha (VRLU).
  4. Using Rényi Upper-Lower Bounds combination as the loss function (VRS).

In addition, it contains implementation of our VRS-MSA model using both DC-programming and SGD.
We used 3 datasets: MNIST, USPS and SVHN.
