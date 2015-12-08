--[[ An implementation of RMSprop

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.alpha'             : smoothing constant
- 'config.epsilon'           : value with which to inistialise m
- 'state'                    : a table describing the state of the optimizer;
                               after each call the state is modified
- 'state.m'                  : leaky sum of squares of parameter gradients,
- 'state.tmp'                : and the square root (with epsilon smoothing)

RETURN:
- dx     : modified gradient

]]
sharpen = require 'spectralDescent.sharp'
approxSharpen = require 'spectralDescent.approxSharp'
local RMSpropsteps={}
--[[
function RMSpropsteps(dfdx)
	return dfdx
end
--]]

function RMSpropsteps(dfdx, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8
    local k = config.k or 30
    
    -- (1) gradient is now an input

    -- (2) initialize mean square values and square gradient storage
    if not state.m then
      state.m = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):zero()
      state.tmp = torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
      state.tmp2 = torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
    end

    -- (3) calculate new (leaky) mean squared values
    state.m:mul(alpha)
    state.m:addcmul(1.0-alpha, dfdx, dfdx)
    
    
    -- (4) perform update
    state.tmp:sqrt(state.m):add(epsilon)
    dfdx=torch.cdiv(dfdx,state.tmp)
    -- df=torch.cdiv(dfdx,state.tmp)
    -- return x*, f(x) before optimization
    return dfdx
end
return RMSpropsteps
