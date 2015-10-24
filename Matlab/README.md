# Stochastic Spectral Descent

This Matlab demo demonstrates the performance of Stochastic Spectral Descent and its variants with local preconditioning, ADAspectral and RMSspectral.

To get started:

1)	For experiments with convolutional neural nets, install MatConvNet: http://www.vlfeat.org/matconvnet/ . 
	Set the path to MatConvNet in the file 'environmentVariables.m'.

2)	Download the MNIST and/or CIFAR10 datasets as specified under the 'data7'.

3)	In 'main.m', specify which experiment to run and whether a GPU device should be used.


The demo will store the resulting models and convergence plots under 'results/'. See the folder for expected output.

