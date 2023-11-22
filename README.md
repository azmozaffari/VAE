# VAE
This is the quick implementation of VAE for generating MNIST images.

The code is written inspired by VAE implementation in https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/Variational_autoencoder.py

The advantage of VAE in comparison to GAN is its smoothness to generate the samples from the Gaussian noise


First train the model by running train.py

Then generate the samples by running test.py 


![Screenshot](image.png)


The facinating part for me in VAE implementation is resampling function which facilitates learning mu and std.

The weekness of VAE comparing to GAN is the quality of the images that is reconstructed which can be improved by adding skip connections between encoder and decoder.

