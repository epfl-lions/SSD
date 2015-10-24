function [err,state] = Adagrad(model,f,opts,state) %--TODO: Add momentum
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


	% Learning rate annealing
    if isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
        lr = lr*(state.iter^-opts.learningRateDecay);
    end
    

	% Call function to update model gradients and return current error
    params = model.getParameters();
    [err, grad] = f();
   
    
	% L2 regularization
    if isfield(opts,'weightDecay') && opts.weightDecay > 0
        grad = grad + opts.weightDecay*params;
    end
    

    % Update history with new gradient
    if ~isfield(state,'history'); state.history = zeros(model.parameterSize,1); end
    state.history = state.history + grad.^2; 
    grad = grad ./ (eps + sqrt(state.history));
    
	% Update model parameters
    params = params - lr*grad;
    model.setParameters(params);
end
