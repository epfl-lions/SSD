%clearvars;
addpath(genpath('.'));
close all
global gpu;
gpu = false;


experiment = 'MNIST';
fprintf('Running %s experiments...\n',experiment)

switch experiment
	case 'MNIST'
		[exps,data,resultsPath] = mnistExperiments();
	case 'CIFAR'
		[exps,data,resultsPath] = cifarConvolutionExperiments();
	otherwise
		disp('Please specify a set of experiments.');
end
	

h = figure;
colors = distinguishable_colors(numel(exps));
legendNames = {};
for ii=1:numel(exps)
    exps{ii}.dataChunkSize = 2000;
    exps{ii}.randSeed = 100;
    exps{ii}.descentOpts.gpu = gpu;
    exps{ii}.reportBatchInterval = 0;
    exps{ii}.run();
    exps{ii}.model.sanitize();
    disp('-----------------');
    set(0,'currentfigure',h);
    x = cumsum(exps{ii}.results.times);
    y = -mean(exps{ii}.results.trainErrors,1);
    line(x,y,'color',colors(ii,:));
	legendNames = [legendNames;exps{ii}.name];
    legend(legendNames{1:ii});
    drawnow;
end
data.clear();
fprintf('Saving...');
save([resultsPath '.mat'], 'exps','-v7.3');


title(experiment)
set(h,'defaultTextInterpreter','latex')
xlabel('Seconds')
ylabel('$\log p(\mathbf{v})$','Interpreter','latex')
ca = get(h,'CurrentAxes');
set(ca,'yscale','log');
print('-dpdf',[resultsPath '.pdf']); 
fprintf(' done!\n');


