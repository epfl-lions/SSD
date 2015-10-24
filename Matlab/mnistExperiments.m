function [exps,data,resultsPath] = mnistExperiments()
	global gpu;
	opts.precision = @single;
	resultsPath = 'results/MNIST-NN';
	fprintf('Experiment results will be saved as %s.mat\n',resultsPath)


	%% Load MNIST data
	opts.flatten = true;
	opts.gpu = gpu;
	Ntrain  = 60000;
	Ntest   = 10000;
	data = MnistData(Ntrain,Ntest,opts);

	%% Define model, in this case a feedforward NN
	sizes = [prod(data.inSize) 300 data.outSize];
	NNFactory = @() SimpleNN(sizes, 'Sigmoid', gpu);

	%% Set up experiments
	exps = {};

	ex = SimpleExperiment('SGD',NNFactory,data,@GradientDescent);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = 0.1;
	ex.descentOpts.learningRateDecay = .2;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;

	ex = SimpleExperiment('ADAgrad',NNFactory,data,@Adagrad);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = .03;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;

	ex = SimpleExperiment('RMSprop',NNFactory,data,@RMSprop);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = .02;
	ex.descentOpts.learningRateDecay = .005;
	ex.descentOpts.RMSpropDecay = .95;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;

	ex = SimpleExperiment('SSD',NNFactory,data,@SpectralDescent);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = 0.2;
	ex.descentOpts.learningRateDecay = .2;
	ex.descentOpts.batchSize = 1500;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;

	ex = SimpleExperiment('ADAspectral',NNFactory,data,@AdaSpectral);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.learningRate = 0.01;
	ex.descentOpts.epsilon = 1e-3;
	ex.descentOpts.batchSize = 1500;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;

	ex = SimpleExperiment('RMSspectral',NNFactory,data,@RMSSpectral);
	ex.descentOpts.gpu = gpu;
	ex.descentOpts.RMSpropDecay = .95;
	ex.descentOpts.learningRate = 0.001;
	ex.descentOpts.learningRateDecay = .15;
	ex.descentOpts.epsilon = 1e-5;
	ex.descentOpts.batchSize = 1500;
	ex.descentOpts.epochs = 50;
	exps{end+1} = ex;
end

