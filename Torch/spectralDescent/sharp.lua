local sharp={}
function sharp(X)
  u,s,v=torch.svd(X)
  Y=torch.mm(u,v[{{},{1,u:size(2)}}]:t()):mul(s:sum())
  return Y
end
return sharp
