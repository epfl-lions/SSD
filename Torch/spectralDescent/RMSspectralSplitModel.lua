RMSspectralsteps=require 'spectralDescent.RMSspectralsteps'
RMSpropsteps=require 'spectralDescent.RMSpropsteps'
local RMSspectralSplitModel
function RMSspectralSplitModel(g, x, config, model, state)

    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-4
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8
    local initialized = state.initialized or false
    -- (0b) initialize split states if necessary
	if initialized==false then
    state.initialized=true
    M=model.modules
    for j=1,#M do
	if torch.type(M[j])=='nn.Linear' then
	     	mat={}
	     	bias={}
	     	st={mat,bias}
	     	table.insert(state,st)
        elseif torch.type(M[j])=='nn.SpatialConvolutionMap' then
     	     	mat={}
  	     	bias={}
	     	st={mat,bias}
	     	table.insert(state,st)
	else
	     	table.insert(state,{})
	end
    	end
	end
    -- (1) evaluate f(x) and df/dx
    	local fx, dfdx = opfunc(x,model)
    -- (2) process model splitting
    

      	local subndx1=1
	local subndx2=0;
	local tmp
	
      for j=1,#M do
        if torch.type(M[j])=='nn.Linear' then
	  subndx2=subndx1+M[j].weight:numel()
 	  tmp=dfdx[{{subndx1,subndx2-1}}]
	  tmp:reshape(tmp,#M[j].weight)
          tmp=sd.RMSspectralsteps(tmp,config,state[j][1])
	  dfdx[{{subndx1,subndx2-1}}]=tmp
	  subndx1=subndx2
	  subndx2=subndx1+M[j].bias:numel()
 	  tmp=dfdx[{{subndx1,subndx2-1}}]
          tmp=sd.RMSpropsteps(tmp,config,state[j][2])
	  dfdx[{{subndx1,subndx2-1}}]=tmp
	  subndx1=subndx2
	elseif torch.type(M[j])=='nn.SpatialConvolutionMap' then
	  subndx2=subndx1+M[j].weight:numel()
 	  tmp=dfdx[{{subndx1,subndx2-1}}]
	  tmp:reshape(tmp,#M[j].weight)
          tmp=sd.RMSpropsteps(tmp,config,state[j][1])
	  dfdx[{{subndx1,subndx2-1}}]=tmp
	  subndx1=subndx2
	  subndx2=subndx1+M[j].bias:numel()
 	  tmp=dfdx[{{subndx1,subndx2-1}}]
          tmp=sd.RMSpropsteps(tmp,config,state[j][2])
	  dfdx[{{subndx1,subndx2-1}}]=tmp
	  subndx1=subndx2     
	end
      end

	--[[
    -- (2) perform state update and get new step direction
      model:updateParameters(opt.learningRate)
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
    local dfdx = RMSspectralsteps(dfdx, config, state)
    local dfdx = RMSpropsteps(dfdx, config, state)
--]]
    -- (3) perform update
    x:add(-lr, dfdx)

    -- return x*, f(x) before optimization
    return x, {fx}
end
return RMSspectralSplitModel
