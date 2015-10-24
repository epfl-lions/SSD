function [err,state] = AdaSpectral(model,f,opts,state) %--TODO: Add momentum
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
		epsilon = opts.epsilon;
	else
		epsilon = 1e-1;
	end


	% Learning rate annealing
	if isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
		lr = lr/(opts.learningRateDecay^state.iter);
	end


	% Call function to update model gradients and return current error
	err = f();


	%%
	L=numel(model.modules);
	if ~isfield(state,'history'); state.history = cell(L,1); end
	for l=1:L
		if isa((model.modules{l}),'Linear')
			% Initialize history
			if isempty(state.history{l})
				siz=size(model.modules{l}.weights);
		        if isa(model.modules{l}.weights,'gpuArray')
		            state.history{l}=gpuArray.zeros(siz+[0 1]);
		        else
		            state.history{l}=zeros(siz+[0 1]);
		        end
		    end

			% Concat weights and bias
		    H = [model.modules{l}.accBiasGrads model.modules{l}.accWeightGrads];

			% L2 regularization
		    if isfield(opts,'weightDecay') && opts.weightDecay > 0
		        H(:,2:end) = H(:,2:end)+ opts.weightDecay*model.modules{l}.weights;
		    end

		    % Update history with new gradient
		    state.history{l} = state.history{l}+ H.^2;
		    history=epsilon+sqrt(state.history{l});
		    hist2=sqrt(history);

			% Apply sharp and rescale
		    k=min(min(size(history)),30);
		    H = lr*projectMatrix(H./hist2,opts.gpu,'approxRandSpectral',k)./hist2;

		    model.modules{l}.accBiasGrads=H(:,1);
		    model.modules{l}.accWeightGrads=H(:,2:end);
		elseif model.modules{l}.parameterSize > 0
			%% Initialize history
			if isempty(state.history{l})
		        if isa(model.modules{l}.weights,'gpuArray')			
			        state.history{l} = gpuArray.zeros(size(model.modules{l}.parameterSize));
				else
			        state.history{l} = zeros(size(model.modules{l}.parameterSize));
				end
			end

			H = model.modules{l}.getParametersGradient();

			%update history with new gradient
			state.history{l} = state.history{l} + H.^2; 
			H = lr*(H./ (epsilon + sqrt(state.history{l})));

			model.modules{l}.setParametersGradient(H);
		end
	end
	model.updateParameters();
end
