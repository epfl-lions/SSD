local randSVD=require 'spectralDescent.randSVD'
local hackdiag=require 'spectralDescent.hackdiag'
local approxSharp={}
function approxSharp(X,k)

  local u,s,v=randSVD(X,k)
  
  local X2=X-torch.mm(u,torch.mm(hackdiag(s),v:t()))
   
  local Y=torch.mm(u,v:t()):mul(s:sum())
  

  --Y:add(s:sum()/(s[-1]+.00001),X-X2)
  return Y
end
return approxSharp
