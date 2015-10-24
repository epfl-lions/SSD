classdef MnistData < handle
    properties
        trainSizeMax = 60000;
        testSizeMax = 10000;
        trainSizeCurrent;
        testSizeCurrent;
        inSize = [28 28 1]; %width height #channels
        outSize = 10;
        trainDataInputsPath = 'data/mnistData/train-images.idx3-ubyte';
        trainDataLabelsPath = 'data/mnistData/train-labels.idx1-ubyte';
        testDataInputsPath = 'data/mnistData/t10k-images.idx3-ubyte';
        testDataLabelsPath = 'data/mnistData/t10k-labels.idx1-ubyte';
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
        function obj = MnistData(Ntrain,Ntest,opts)
            
            
            if nargin<1; Ntrain  = obj.trainSizeMax; end
            if nargin<2; Ntest   = obj.testSizeMax; end
            if nargin > 2; obj.readOptions(opts); end
            
            obj.trainSizeCurrent = min(Ntrain,obj.trainSizeMax);
            obj.testSizeCurrent = min(Ntest,obj.testSizeMax);
            
            % Training set files
            files = cell(1);
            files{1}.inputs = obj.trainDataInputsPath;
            files{1}.labels = obj.trainDataLabelsPath;
            obj.train = Dataset(obj,'train',files);
            
            % Test set files
            files = cell(1);
            files{1}.inputs = obj.testDataInputsPath;
            files{1}.labels = obj.testDataLabelsPath;
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
        
        
        function batch = loadDataFile(obj,paths)
            batch.inputs = loadMNISTImages(paths.inputs);
            batch.labels = loadMNISTLabels(paths.labels)+1;
        end
        
        function preprocessData(obj,dataset)
            dataset.inputs = obj.precision(dataset.inputs);          
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
