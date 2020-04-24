# Variational Autoencoder(VAE)

## Training
```shell
cd vision/vae_mnist
julia --project vae_mnist.jl
```

Original iamges

![Original](docs/original.png)

10 epochs

![10 epochs](docs/epoch_10.png)

20 epochs

![10 epochs](docs/epoch_20.png)

## Visualization
```shell
julia --project vae_plot.jl
```
Visualization of latent space

![Clustering](docs/clustering.png)

Visualization of 2D manifold

![Manifold](docs/manifold.png)