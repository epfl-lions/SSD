sharpen = require 'spectralDescent.sharp'
approxSharpen = require 'spectralDescent.approxSharp'
local RMSspectralsteps
function RMSspectralsteps(dfdx, config, state)
    -- (0) get/update state

    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-4
    local alpha = config.alpha or 0.95
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
    state.tmp:sqrt(state.m):add(epsilon):sqrt()
    state.tmp2:cdiv(dfdx,state.tmp)
    if (k==0) or (k>=dfdx:size(1)) or (k>=dfdx:size(2)) then
    state.tmp2=sharpen(state.tmp2)
    else
    state.tmp2=approxSharpen(state.tmp2,k)
  end

    state.tmp2:cdiv(state.tmp)
    -- sharp soon
    dfdx=state.tmp2
    -- return x*, f(x) before optimization
    return dfdx
end
return RMSspectralsteps
