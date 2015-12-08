# Stochastic Spectral Descent

This Matlab demo demonstrates the performance of RMSspectral.

To get started:

To run on the GPU, this requires the compilation of torch with Magma installed (http://icl.cs.utk.edu/magma/).  We used Magma version 1.7.  These experiments give very similar results to the MATLAB code.

To run RMSspectral:
th spectralDescentCNN.lua --optimization 'RMSspectral'

To run RMSprop:
th spectralDescentCNN.lua --optimization 'RMSprop'

To run SGD:
th spectralDescentCNN.lua --optimization 'SGD'

Note that the default batch size is large for RMSspectral.  The batch size can be reduced for RMSprop and SGD.

