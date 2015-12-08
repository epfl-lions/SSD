
local randSVD={}
function randSVD(Xin,k)
        local trans=false
        local X
        if Xin:size(2) > Xin:size(1) then
            trans=true
            X=Xin:t()
          else
            
            X=Xin
        end
        
  local omeg=sddefault()
  omeg:randn(X:size(2),k)
  local Y=X*omeg
  local q,r=torch.qr(Y)
  local B=q:t()*X
  local uhat,s,v=torch.svd(B)
  local v=v[{{},{1,k}}]
  local u=q*uhat
  -- return u,s,v

  if trans then
      return v,s,u
    else
  return u,s,v
  end
  
end
return randSVD
