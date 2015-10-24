function [exps,data,resultsPath] = cifarConvolutionExperiments()
	global gpu;
	opts.precision = @single;
	resultsPath = 'results/CIFAR-CNN';
	fprintf('Experiment results will be saved as %s.mat\n',resultsPath)

	%% Load Cifar10 data
	opts.flatten = false;
	opts.gpu = gpu;
	opts.whiten = true;
	Ntrain  = 50000;
	Ntest   = 10000;
	data = Cifar10Data(Ntrain,Ntest,opts);


	%% Define model, in this case a convolutional NN
	% convolutionSizes = ...
	% [5       5  ;       % filter height
	%  5       5  ;       % filter width
	%  16      16 ;       % #out channels 
	%  2       2  ;       % max pooling height
	%  2       2  ]';     % max pooling width
	% %1st    2nd           layers

	convolutionSizes = [5 5 32 2 2];
	linearSizes = [300 data.outSize]; 
	NNFactory = @() SimpleCNN(data.inSize,convolutionSizes,linearSizes,'ReLU', gpu);


	%% Set up experiments
	exps = {};

	ex = SimpleExperiment('SGD',NNFactory,data,@GradientDescent);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = 0.05;
	ex.descentOpts.learningRateDecay = .2;
	ex.descentOpts.batchSize = 100;
	ex.descentOpts.epochs = 35;
	exps{end+1} = ex;

	ex = SimpleExperiment('ADAgrad',NNFactory,data,@Adagrad);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = .02;
	ex.descentOpts.batchSize = 100;
	ex.descentOpts.epochs = 35;
	exps{end+1} = ex;

	ex = SimpleExperiment('RMSprop',NNFactory,data,@RMSprop);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = .01;
	ex.descentOpts.learningRateDecay = .01;
	ex.descentOpts.RMSpropDecay = .9;
	ex.descentOpts.batchSize = 100;
	ex.descentOpts.epochs = 35;
	exps{end+1} = ex;

	ex = SimpleExperiment('SSD',NNFactory,data,@SpectralDescent);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = 0.005;
	ex.descentOpts.learningRateDecay = .01;
	ex.descentOpts.batchSize = 500;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;

	ex = SimpleExperiment('ADAspectral',NNFactory,data,@AdaSpectral);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = 0.01;
	ex.descentOpts.epsilon = 0.05;
	ex.descentOpts.batchSize = 500;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;

	ex = SimpleExperiment('RMSspectral',NNFactory,data,@RMSSpectral);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = 0.01;
	ex.descentOpts.learningRateDecay = .01;
	ex.descentOpts.RMSpropDecay = .9;
	ex.descentOpts.epsilon = 0.05;
	ex.descentOpts.batchSize = 500;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;
end
