function [err,state] =RMSSpectral3(model,f,opts,state) %--TODO: Add momentum
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

	if isfield(opts,'RMSpropDecay')
		rmsd = opts.RMSpropDecay;
	else
		rmsd = 0.99;
	end


	% Learning rate annealing
	if isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
		lr = lr*(state.iter^-opts.learningRateDecay);
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
		            state.history{l}=gpuArray.ones(siz+[0 1]);
		        else
		            state.history{l}=ones(siz+[0 1]);
		        end
		    end

			%concat weights and bias
		    H = [model.modules{l}.accBiasGrads model.modules{l}.accWeightGrads];

			%L2 regularisatzation
		    if isfield(opts,'weightDecay') && opts.weightDecay > 0
		        H(:,2:end) = H(:,2:end)+ opts.weightDecay*model.modules{l}.weights;
		        if opts.reportL2Penalty; err = err + 0.5*opts.weightDecay*dot(model.modules{l}.weights(:),model.modules{l}.weights(:)); end
		    end
	
			%update history with new gradient
		    state.history{l} = rmsd*state.history{l}+ (1-rmsd)*H.^2;
		    history=eps+sqrt(state.history{l});
		    hist2=sqrt(history);
		    k=min(min(size(history)),30);

			%apply sharp and rescale
		    H = lr*projectMatrix(H./hist2,opts.gpu,'approxRandSpectral',k)./hist2;

			%split the concatenated gradient
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
			params = model.modules{l}.getParameters();

			if isfield(opts,'weightDecay') && opts.weightDecay > 0
				H = H + opts.weightDecay*params;
				if opts.reportL2Penalty; err = err + 0.5*opts.weightDecay*dot(params(:),params(:)); end
			end


			% Initialize and update history with new gradient
			if isempty(state.history{l}); state.history{l} = 1+H.^2; end
			state.history{l} = rmsd*state.history{l} + (1-rmsd)*H.^2;

		
			H = lr*(H./(eps + sqrt(state.history{l})));
			model.modules{l}.setParametersGradient(H);
		end
	end
	model.updateParameters();
end
