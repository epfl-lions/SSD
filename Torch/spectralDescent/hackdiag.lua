local hackdiag={}
function hackdiag(s)
	l=s:size(1)
	--X=torch.CudaTensor()
	--X=torch.DoubleTensor()
	X=sddefault()	
	X:zeros(l,l)
	--X=torch.zeros(l,l)
	for c=1,l do
		X[c][c]=s[c]
	end
  return X
end
return hackdiag
