RMSspectralsteps=require 'spectralDescent.RMSspectralsteps'
RMSpropsteps=require 'spectralDescent.RMSpropsteps'
local RMSspectral
function RMSspectral(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- (2) perform state update and get new step direction
    local dfdx = RMSspectralsteps(dfdx, config, state)
  --  local dfdx = RMSpropsteps(dfdx, config, state)
    -- (3) perform update
    x:add(-lr, dfdx)

    -- return x*, f(x) before optimization
    return x, {fx}
end
return RMSspectral
