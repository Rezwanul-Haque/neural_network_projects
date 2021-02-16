# What is autoencoders?

Autoencoders are neural networks that learn a compressed representation of the input, known as the latent
representation. They are different from conventional feed forward neural networks because their structure consists of an
encoder and a decoder component, which is not present in CNNs.

# Application of autoencoders

1. Image denoising
2. Noisy documents denoising

# Image Denoising

Image noise is defined as a random variations of brightness in an image. Image noise may originate from the sensors of
digital cameras. Although digital cameras these days are capable of capturing high quality images, image noise may still
occur, especially in low light conditions

# What happens when the latent representation of the autoencoder is too small?

The size of the latent representation should be sufficiently small enough to represent a compressed representation of
the input, and also be sufficiently large enough for the decoder to reconstruct the original image without too much
loss.

# What are the input and output when training a denoising autoencoder?

The input to a denoising autoencoder should be a noisy image and the output should be a reference clean image. During
the training process, the autoencoder learns that the output should not contain any noise
(through the loss function), and the latent representation of the autoencoder should only contain the signals (that is,
non-noise elements)

# What are some of the ways we can improve the complexity of denoising autoencoders?

For denoising autoencoders, convolutional layers always work better than dense layers, just as CNNs work better than
conventional feed forward neural networks for image classification tasks. We can also improve the complexity of our
model by building a deeper network with more layers, and by using more filters in each convolutional layer.

