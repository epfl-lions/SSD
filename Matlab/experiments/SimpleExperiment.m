classdef SimpleExperiment < Experiment
    properties  
        descentOpts;
        results = {};
        duration;
        saveInterval = -1;
        trainError=[];
        optim;
        optimState = {};
        reportBatchInterval = 50;
        reportEpochInterval = 1;
        dataChunkSize = 3000;
        reloaded = false;
        times = [];
    end
    methods
        function obj = SimpleExperiment(name, modelFactory, data, optim)
            obj = obj@Experiment(name, modelFactory, data);
            obj.descentOpts.momentum = 0;
            obj.descentOpts.learningRateDecay = 0;
            obj.descentOpts.L2WeightDecay = 0;
            obj.descentOpts.learningRate = 0.1;
            obj.descentOpts.gpu = false;
            obj.descentOpts.initialEpoch = 1;
            obj.descentOpts.epochs = 10;
            obj.descentOpts.batchSize = 100;           
            obj.optim = optim;
            
        end
        
        
        function runExperiment(obj)
            disp(obj.name)
            trainErrors = [];
            testAccuracy = [];
			if obj.descentOpts.gpu; gd = gpuDevice(); end
            times=[];
            for epoch=obj.descentOpts.initialEpoch:obj.descentOpts.epochs
                rng(obj.randSeed+epoch,'twister');
	            tic;
                trainErrors = [trainErrors  obj.trainEpoch(obj.model,obj.data.train.getIterator(obj.dataChunkSize))'];
				if obj.descentOpts.gpu; wait(gd); end
				times=[times;toc];
%                 trainAccuracy = [trainAccuracy ;obj.evaluate(obj.model,obj.data.train)];
                testAccuracy = [testAccuracy ; obj.evaluate(obj.model,obj.data.test.getIterator(obj.dataChunkSize))];
                if obj.saveInterval > 0 && mod(epoch,obj.saveInterval) == 0
                    obj.save();
                end
				
                if obj.reportEpochInterval > 0 && mod(epoch,obj.reportEpochInterval) == 0
                    fprintf('%s, epoch %d error: %g',obj.name,epoch,mean(trainErrors(:,end)));
                    fprintf(', test set accuracy: %g',testAccuracy(end));
                    fprintf(', duration: %g',sum(times(end)));
                    fprintf('\n');
                end
            end
            
            if isfield(obj.results,'trainErrors')
                obj.results.trainErrors = [obj.results.trainErrors trainErrors];
            else
                obj.results.trainErrors = trainErrors;
            end
            
            trainAccuracy = obj.evaluate(obj.model,obj.data.train.getIterator(obj.dataChunkSize));
            if isfield(obj.results,'trainAccuracy')
                obj.results.trainErrors = [obj.results.trainAccuracy ; trainAccuracy];
            else
                obj.results.trainAccuracy = trainAccuracy;
            end
  
            if isfield(obj.results,'testAccuracy')
                obj.results.testAccuracy = [obj.results.testAccuracy testAccuracy];
            else
                obj.results.testAccuracy = testAccuracy;
            end

            if isfield(obj.results,'times')
                obj.results.times = [obj.results.times ; times];
            else
                obj.results.times = times;
            end            
            
%             obj.results.testAccuracy = obj.evaluate(obj.model,obj.data.test);
            disp(obj.name)
            fprintf('Training set accuracy: %g\n',obj.results.trainAccuracy);
            fprintf('Test set accuracy: %g\n',obj.results.testAccuracy(end));
        end
        
        
        function errs = trainEpoch(obj, model, data)
            criterion = NLLCriterion();
            errs = [];
            batch = {};
            batch.number = 0;
            
            function [fx,dfdx] = f(x)
                if nargin>1 && ~isempty(x); model.setParameters(x);end
                
                preds = model.forward(batch.inputs);
                fx = criterion.forward(preds,batch.labels)/(hi-lo+1);
                if obj.reportBatchInterval > 0 && mod(batch.number,obj.reportBatchInterval) == 0
                    fprintf('Batch error: %g\n',fx);
                end
                
                grads = criterion.backward(preds,batch.labels);
                grads = grads./(hi-lo+1); % average according to batch size
                model.backward(batch.inputs,grads);
                
                if nargout > 1; dfdx = model.getParametersGradient(); end
            end
            
            dataChunk = data();
            while dataChunk.size > 0
                %fprintf('Chunk %d, from %d to %d -> %d samples.\n',dataChunk.ind,dataChunk.lo,dataChunk.hi,dataChunk.size);
                ind(1:ndims(dataChunk.inputs)) = {':'};
                for lo = 1:obj.descentOpts.batchSize:dataChunk.size
                    batch.number = batch.number+1;
                    hi = min(lo+obj.descentOpts.batchSize-1,dataChunk.size);
                    ind{end} = lo:hi;%shuffle(lo:hi);
                    batch.inputs = dataChunk.inputs(ind{:});
                    batch.labels = dataChunk.labels(lo:hi);
                    if obj.reloaded
                        obj.reloaded = false;
                    else
                        model.setParametersGradient(zeros(model.parameterSize,1));
                    end
                    %tic
                    [err, obj.optimState] = obj.optim(obj.model,@f,obj.descentOpts,obj.optimState);
                    %obj.times(end+1) = toc;
                    errs = [errs err];
                end
                dataChunk = data();
            end
        end
        
        function acc = evaluate(obj,model,data)
            acc = 0;
            dataChunk = data();
            while dataChunk.size > 0
                preds = model.forward(dataChunk.inputs);
                [~, maxInd] = max(preds);
                acc = acc+sum(maxInd'==dataChunk.labels);
                dataChunk = data();
            end
            acc = acc/dataChunk.dataSize;
        end
        
        
        
        function sobj = saveobj(obj)
            sobj = saveobj@Experiment(obj);
            sobj.data.clear();
            sobj.model.sanitize();
            sobj.descentOpts = obj.descentOpts;
            sobj.optim = obj.optim;
            sobj.optimState = obj.optimState;
            sobj.results = obj.results;
        end
        
        function obj = reload(obj,sobj)
            obj = reload@Experiment(obj,sobj);
            obj.descentOpts = sobj.descentOpts;
            obj.optim = sobj.optim;
            obj.optimState = sobj.optimState;
            obj.results = sobj.results;
            obj.reloaded = true;
        end
        
    end
    methods (Static)
         function obj = loadobj(sobj)
            obj = SimpleExperiment(sobj.name,sobj.modelFactory,sobj.data,sobj.optim);
            obj.reload(sobj);
        end
    end
    
end
