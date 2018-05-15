function [modelParameters] = positionEstimatorTraining(training_data)
% Arguments:

% - training_data:
%     training_data(n,k)              (n = trial id,  k = reaching angle)
%     training_data(n,k).trialId      unique number of the trial
%     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

% ... train your model

% Return Value:

% - modelParameters:
%     single structure containing all the learned parameters of your
%     model and which can be used by the "positionEstimator" function.

  
%% Data preparation  
m = 300; % ms analysed
z = 0;
u = 50; % # of neurons in the sum
norm = 1;
modelParameters{3} = m;
modelParameters{4} = z;
modelParameters{5} = u;
modelParameters{6} = norm;

l = 0; % max # of datapoints possible to create
q = 1;
 
% calculate max # of datapoints possible to create
for n=1: length(training_data(:,1)) % training data: 80% of all trials
    for k=1:8 % reaching angles
        s = 320;
        while(1)
            l = l + 1;
            s = s + 20; % step 20ms
            if s-20==length(training_data(n,k).spikes(1,:))-120
                break;
            elseif s>length(training_data(n,k).spikes(1,:))-120
                s = length(training_data(n,k).spikes(1,:))-120;
            end
        end
    end
end
 
x = zeros(98*(m-z)/u, l); % training data
x_temp = [];
t = zeros(8, l); % training labels
t_temp = [];
 
for n=1:length(training_data(:,1))
    for k=1:8
        s = 320;
        while(1)
            % crate one datapoint 98x6
            for i=m:-u:(z+1) 
                sum = zeros(98,1);
                % average over 50ms
                for j=0:(u-1)
                    sum = sum + training_data(n,k).spikes(:,s-i+j);
                end
                if norm == 1
                    x_temp = [x_temp; sum/u*14];
                else
                    x_temp = [x_temp; sum];
                end
            end
            x(:,q) = x_temp;
            x_temp = [];
 
            % create training labels
            t_temp = zeros(8, 1);
            t_temp(k) = 1; % for one angle
            t(:,q) = t_temp;
            t_temp = [];
 
            q = q + 1;
            s = s + 20; % step 20ms
            if s-20==length(training_data(n,k).spikes(1,:))-120
                break;
            elseif s>length(training_data(n,k).spikes(1,:))-120
                s = length(training_data(n,k).spikes(1,:))-120;
            end
        end
    end
end
  
%% Neural Network
rep = 3; % number of neural nets in committee machine
modelParameters{7} = rep;
net_tab = cell(1,rep);

% Train the Network
for n=1:rep
    n
    % Choose a Training Function
    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

    % Create a Pattern Recognition Network
    hiddenLayerSize = [100];
    net = patternnet(hiddenLayerSize, trainFcn);
    
    if mod(n,2)
        net.layers{1}.transferFcn = 'tansig';
    else
        net.layers{1}.transferFcn = 'radbas';
    end
    
%     % number of valodation checks
%     net.trainParam.max_fail = 4;

    % disable graphic output
    net.trainParam.showWindow = false;


    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    % Train the Network
    [net,~] = train(net,x,t);
    net_tab{n} = net;
    clear net
end
  

%% Average trajectory
% calculate the longest trial
m = 0;
for i=1:length(training_data(:,1))
    for j=1:length(training_data(1,:))
        l = length(training_data(i,j).spikes(1,:));
        if l>m
            m = l;
        end
    end
end

% average for each angle over all trials
avg_trj(length(training_data(1,:))).handPos = [];

for j=1:length(training_data(1,:))
    trj = zeros(3,m);
    for i=1:length(training_data(:,1))
        for k=1:length(training_data(i,j).handPos(1,:))
            trj(:,k) = trj(:,k) + training_data(i,j).handPos(:,k);
        end
        for h=k:m
            trj(:,h) = trj(:,k);
        end
    end
    avg_trj(j).handPos=trj(:,:)/length(training_data(:,1));
end

% save parameters
modelParameters{1} = net_tab;
modelParameters{2} = avg_trj;
end