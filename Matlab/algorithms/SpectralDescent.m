function [err,state] =SpectralDescent(model,f,opts,state) %--TODO: Add momentum
	% Initialize state
	if nargin < 3; state = {}; end
	if ~isfield(state,'iter'); state.iter = 0; end
	state.iter = state.iter+1;


	% Set default parameter values
	if isfield(opts,'learningRate')
		lr = opts.learningRate;
	else
		lr = 1;
	end

	if isfield(opts,'epsilon')
		eps = opts.epsilon;
	else
		eps = 1e-1;
	end


	% Call function to update model gradients and return current error
	err = f();


	% Learning rate annealing
	if isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
		lr = lr*(state.iter^-opts.learningRateDecay);
	end


	%%
	L=numel(model.modules);
	for l=1:L
		if isa(model.modules{l},'Linear') % Operate on linear layers only
			% Concat weights and bias
		    H = [model.modules{l}.accBiasGrads model.modules{l}.accWeightGrads];

			% L2 regularization
		    if isfield(opts,'weightDecay') && opts.weightDecay > 0
		        H(:,2:end) = H(:,2:end)+ opts.weightDecay*model.modules{l}.weights;
		    end

			%apply sharp and rescale
		    k=min(min(size(model.modules{l}.weights)),30);
		    H = lr*projectMatrix(H,opts.gpu,'approxRandSpectral',k);

			%split the concatenated gradient
		    model.modules{l}.accBiasGrads=H(:,1);
		    model.modules{l}.accWeightGrads=H(:,2:end);
		end
	end

	% Update model parameters
	model.updateParameters();
end
