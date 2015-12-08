RMSspectralsteps=require 'spectralDescent.RMSspectralsteps'
RMSpropsteps=require 'spectralDescent.RMSpropsteps'
local RMSspectralSplitModelSteps
function RMSspectralSplitModelSteps(dfdx, config, model, state)

    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local alpha = config.alpha or 0.95
    local epsilon = config.epsilon or 1e-8
    local initialized = state.initialized or false
    -- (0b) initialize split states if necessary
	if initialized==false then
    state.initialized=true
    M=model.modules[1].modules
    print('Initializing structures for RMSspectral')
    for j=1,#M do
      print(torch.type(M[j]))
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
    -- (1) process model splitting
    

      	local subndx1=1
	local subndx2=0;
	local tmp
	
      for j=1,#M do
        if torch.type(M[j])=='nn.Linear' then
	  subndx2=subndx1+M[j].weight:numel()
 	  tmp=dfdx[{{subndx1,subndx2-1}}]
	  tmp:reshape(tmp,#M[j].weight)
          tmp=sd.RMSspectralsteps(tmp,config,state[j][1])
          --tmp=sd.RMSpropsteps(tmp,config,state[j][1])
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

    return dfdx
end
return RMSspectralSplitModelSteps
