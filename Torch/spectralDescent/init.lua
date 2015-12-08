
local spectralDescent = {}
sddefault=torch.DoubleTensor;
spectralDescent.sharp = require 'spectralDescent.sharp'
spectralDescent.randSVD = require 'spectralDescent.randSVD'
spectralDescent.RMSpropsteps = require 'spectralDescent.RMSpropsteps'
spectralDescent.RMSspectralsteps = require 'spectralDescent.RMSspectralsteps'
spectralDescent.RMSspectral = require 'spectralDescent.RMSspectral'
spectralDescent.RMSspectralSplitModel = require 'spectralDescent.RMSspectralSplitModel'
spectralDescent.RMSspectralSplitModelSteps = require 'spectralDescent.RMSspectralSplitModelSteps'
return spectralDescent
