function x=projectMatrix(x,gpu,projectionName,xnorm)
	if nargin<3;projectionName='spectral';end
	if nargin<4
		switch projectionName
		    case 'linear'
		        x=x;
		    case 'spectraltrue'
		        x=projectSpectralTrue(x,gpu);
		    case 'approxRandSpectral'
		        k=30;
		        x=projectApproxRandSpectral(x,gpu,k);
		    case 'L1row'
		        x=projectL1row(x,gpu);
		    otherwise
		        error('Not a used norm.')
		end
	else
		switch projectionName
		    case 'linear'
		        x=x;
		    case 'spectraltrue'
		        x=projectSpectralTrue(x,gpu,xnorm);
		    case 'approxRandSpectral'
   		        k=xnorm;
		        x=projectApproxRandSpectral(x,gpu,k);
		    otherwise
		        error('Not a used norm.')
		end
	end
end

%% Infinity norm sharp operator with exact SVD
function x=projectSpectralTrue(x,gpu,mS)
	[U,S,V]=svd(x,'econ');
	S=diag(S);
	if nargin<3
		mS=sum(S);
	end
	r=sum(S>mS*1e-8);
	if gpu
		one2r = gpuArray.colon(1, r);
		x=U(:,one2r)*V(:,one2r)'*mS;
	else
		x=U(:,1:r)*V(:,1:r)'*mS;
	end
end

%% Infinity norm sharp operator with approximate SVD
function P=projectApproxRandSpectral(x, gpu, k)
	[U,S,V] = randomizedSVD(x,k,k+10,3);
	S=diag(S);
	mS=sum(S);
	r=sum(S>mS*1e-8);
	if gpu
		one2r = gpuArray.colon(1, r);
		P=U(:,one2r)*V(:,one2r)'*mS;
	else
		P=U(:,1:r)*V(:,1:r)'*mS;
	end
end



%% Row L1-norm sharp operator
function P=projectL1row(x, gpu, ~)
	rows=1:size(x,1);
	[mvals,idx] = max(abs(x),[],2);
	idx = sub2ind(size(x),rows(:),idx(:));
	P = zeros(size(x));
	if gpu;P=gpuArray(P);end
	P(idx) = sum(mvals)*sign(x(idx));
end






