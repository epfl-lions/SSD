classdef Cifar10Data < handle
    properties
        numBatches = 5;
        trainSizeMax = 50000;
        testSizeMax = 10000;
        trainSizeCurrent;
        testSizeCurrent;
        inSize = [32 32 3]; %width height #channels
        outSize = 10;
        trainDataPath = 'data/cifar10Data/data_batch_%d.mat';
        testDataPath = 'data/cifar10Data/test_batch.mat';
        precision = @double;
        flatten = true;
        gpu = false;
        whiten = true;
        train;
        test;
        hasData = false;
        mean;
        whM;
        whMInv;
    end
    methods
        function obj = Cifar10Data(Ntrain,Ntest,opts)
            
            
            if nargin<1; Ntrain  = obj.trainSizeMax; end
            if nargin<2; Ntest   = obj.testSizeMax; end
            if nargin > 2; obj.readOptions(opts); end
            
            obj.trainSizeCurrent = min(Ntrain,obj.trainSizeMax);
            obj.testSizeCurrent = min(Ntest,obj.testSizeMax);
            
            % Training set files
            files = cell(obj.numBatches,1);
            for ii=1:obj.numBatches
                files{ii} = sprintf(obj.trainDataPath,ii);
            end
            obj.train = Dataset(obj,'train',files);
            
            % Test set files
            files = {obj.testDataPath};
            obj.test = Dataset(obj,'test',files);
            
        end
        
        function load(obj)          
            if ~obj.hasData

                obj.train.loadDataToRAM(obj.trainSizeCurrent);
                obj.test.loadDataToRAM(obj.testSizeCurrent);
                
                obj.hasData = true;
            end
        end
        
        
        function iter = getIterator(obj,data)
            iter = @iterator;
            function chunk = iterator(n)
                n = min(n,numel(data));
                chunk = data{n};
                if obj.gpu; chunk.inputs = gpuArray(chunk.inputs); end
                
                if ~obj.flatten
                    chunk.inputs = reshape(chunk.inputs, [obj.inSize size(chunk,2)]);
                end
            end
        end
        
        
        function clear(obj)
            obj.train = [];
            obj.test = [];
            obj.hasData = false;
        end
        
        
        function batch = loadDataFile(obj,path)
            data = load(path);
            batch.inputs = data.data;
            batch.labels = data.labels+1;
        end
        
        function preprocessData(obj,dataset)
            dataset.inputs = obj.precision(dataset.inputs')/255;
            if obj.whiten
                warning('off','MATLAB:warn_r14_stucture_assignment');
                if strcmp(dataset.name,'train')
                    [dataset.inputs,obj.mean,obj.whMInv,obj.whM] = whiten(dataset.inputs);
                else %test
                    dataset.inputs = obj.whM*bsxfun(@minus, dataset.inputs, obj.mean);
                end
            end            
        end

        function chunk = processChunk(obj,chunk)
            if chunk.size > 0
                if obj.gpu
                    chunk.inputs = gpuArray(chunk.inputs);
                    chunk.labels = gpuArray(chunk.labels);
                end

                if ~obj.flatten
                    chunk.inputs = reshape(chunk.inputs,[obj.inSize numel(chunk.labels)]);
                end
            end
            chunk.inSize = obj.inSize;
        end
        
        function readOptions(self,opts)
            fields = fieldnames(opts);
            for ii=1:numel(fields)
                f = fields{ii};
                if isprop(self,f)
                    self.(f) = opts.(f);
                end
            end
        end  
    end
end

