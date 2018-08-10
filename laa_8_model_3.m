function [V, phi, th] = model_3(x)

x1 = x(:,1);
x2 = x(:,2);
x3 = x(:,3);
x4 = x(:,4);
x5 = x(:,5);
x6 = x(:,6);
x7 = x(:,7);
x8 = x(:,8);
x9 = x(:,9);
x10 = x(:,10);
x11 = x(:,11);
x12 = x(:,12);
x13 = x(:,13);
x14 = x(:,14);
x15 = x(:,15);
x16 = x(:,16);
x17 = x(:,17);
x18 = x(:,18);
x19 = x(:,19);
x20 = x(:,20);
x21 = x(:,21);
x22 = x(:,22);
x23 = x(:,23);
x24 = x(:,24);

phi = [ ( ( (((x1*1.0 * 1.68037968 + x2*-0.10958589 * 0.94430809 + x3*-6.14159265 * 1.0 + x4*-5.54647909 * 1.0 + x5*6.14159265 * 1.0 + x6*0.00618956 * 1.0 + x7*23.72497216 * 1.0 + x8*5.80825932 * 1.0 + x9*3.14159265 * 1.0 + x10*-6.29312138 * 1.67369359 + x11*-9.12526225 * 1.0 + x12*-9.12526225 * 0.33735589 + x13*0.61111111 * 0.52984601 + x14*0.49381044 * 1.0 + x15*5.44139809 * 0.721804 + x16*0.49381044 * 1.0 + x17*9.012196 * 1.0 + x19*-3.32794647 * 1.0 + x20*-3.0 * 0.91794375 + x21*0.14901868 * 0.45691675 + x22*1.0 * 1.0 + x23*2.0 * 1.0 + x24*-0.61111111 * 1.0 + 0.33333333 * 1.0)) .* ((x1*2.0 * 0.91730581 + x2*-9.81811107 * 1.0 + x3*23.72497216 * 1.26753342 + x4*3.0 * 1.0 + x5*5.44139809 * 1.43853209 + x6*-0.10958589 * 0.22044527 + x7*11.31525049 * 0.55037955 + x8*9.012196 * 1.15991627 + x9*-6.51838556 * 1.0 + x10*-9.12526225 * 1.0 + x11*12.07511059 * 2.36297894 + x12*-9.12526225 * 1.0 + x13*5.44139809 * 1.0 + x14*-5.54647909 * 1.0 + x15*-0.61111111 * 1.0 + x16*0.24384875 * 1.0 + x17*-6.14159265 * 1.0 + x18*-1.0 * 1.0 + x19*1.0 * 1.35854634 + x20*2.5 * 1.0 + x21*0.16 * 1.0 + x22*-4.74341649 * 1.0 + x23*6.51838556 * 1.0 + x24*6.6028795 * 1.0 + -2.96825397 * 1.14797242)))  -  power(((x23*1.0 * 1.0)), 3) ) ) , ...
        ( ( sin(((x23*1.0 * 1.0)))  +  ( sin(((x9*1.0 * 1.0)))  .*  (((x4*1.0 * 1.0)) .*  (((x1*2.0 * 1.0 + x2*-4.87082869 * 1.83152033 + x3*5.44139809 * 1.0 + x4*-5.80825932 * 1.20558887 + x5*-5.80825932 * 0.60284229 + x6*6.51838556 * 1.0 + x7*4.87082869 * 1.0 + x8*5.0 * 1.0 + x9*-1.84882636 * 1.0 + x10*-0.26395278 * 0.62470019 + x11*-3.36381487 * 1.0 + x12*5.51838556 * 1.0 + x13*-0.33333333 * 3.32378328 + x14*0.10537254 * 1.0 + x15*-6.51838556 * 1.0 + x16*1.0 * 1.0 + x17*-0.23900967 * 1.26600525 + x18*0.24384875 * 1.0 + x19*-9.12526225 * 1.0 + x20*-3.0 * 1.03435718 + x21*2.35509641 * 1.0 + x22*1.5 * 1.32321385 + x23*5.51838556 * 0.19569839 + -9.12526225 * 0.37317824)) - ((x2*11.31525049 * 1.0 + x3*-5.54647909 * 2.60098224 + x4*9.01838556 * 1.0 + x6*-1.49381044 * 1.10726237 + x7*9.01838556 * 0.27046345 + x8*-2.14159265 * 1.0 + x9*9.01838556 * 0.71501161 + x10*9.012196 * 1.0 + x11*3.5 * 1.0 + x12*-6.29312138 * 1.0 + x13*1.5 * 1.0 + x14*-1.0 * 0.73254573 + x15*5.44139809 * 1.0 + x18*-4.74341649 * 1.0 + x19*3.14159265 * 1.0 + x20*0.16 * 0.27896588 + x21*0.5 * 1.51201825 + x22*1.73205081 * 1.0752844 + x23*4.87082869 * 1.0 + x24*0.11111111 * 1.0 + 4.87082869 * 0.62251133))) ) ) ) ) , ...
        ( (((x1*0.61111111 * 0.82809355 + x2*-1.0 * 1.0 + x4*2.35509641 * 1.24792344 + x5*9.012196 * 1.0 + x6*23.72497216 * 1.0 + x7*-6.29312138 * 1.0 + x8*27.21086426 * 1.0 + x9*-9.12526225 * 1.0 + x10*-6.51838556 * 1.29921126 + x11*-3.14159265 * 1.0 + x12*0.5 * 1.0 + x13*5.44139809 * 1.0 + x14*0.33333333 * 0.99555238 + x15*-0.5 * 1.03456966 + x16*1.0 * 2.08270128 + x17*0.4 * 1.0 + x18*-1.0 * 1.0 + x19*-3.73887877 * 1.0 + x20*-0.49381044 * 1.0 + x22*6.6028795 * 1.0 + x23*-0.5 * 1.0 + x24*3.14159265 * 1.0 + 5.51838556 * 1.0)) .* 0.00330919) ) , ...
        (((x1*0.24384875 * 0.64107464 + x2*3.14159265 * 1.0 + x3*11.31525049 * 1.0 + x4*-9.12526225 * 1.10293099 + x5*3.14159265 * 1.0 + x6*-1.49381044 * 1.0 + x7*4.87082869 * 1.0 + x8*0.5 * 1.0 + x9*-5.54647909 * 1.0 + x10*-3.47492598 * 1.01016464 + x11*3.14159265 * 1.01397657 + x12*-3.47492598 * 1.0 + x13*0.00618956 * 1.0 + x14*0.33333333 * 1.0 + x15*2.0 * 1.01509054 + x16*1.0 * 1.80564591 + x17*2.35509641 * 1.0 + x18*0.33333333 * 1.0 + x19*1.49381044 * 1.0 + x20*0.24384875 * 1.0417788 + x21*-1.0 * 1.0 + x22*1.0 * 1.0 + x24*0.5 * 1.0 + 2.35509641 * 1.0))) , ...
        ( power( ( (((x1*0.61111111 * 0.82809355 + x2*-1.0 * 1.0 + x4*2.35509641 * 1.24792344 + x5*9.012196 * 1.0 + x6*23.72497216 * 1.0 + x7*-6.29312138 * 1.0 + x8*27.21086426 * 1.0 + x9*-9.12526225 * 1.0 + x10*-6.51838556 * 1.29921126 + x11*-3.14159265 * 1.0 + x12*0.5 * 1.0 + x13*5.44139809 * 1.0 + x14*0.33333333 * 0.99555238 + x15*-0.5 * 1.03456966 + x16*1.0 * 2.08270128 + x17*0.4 * 1.0 + x18*-1.0 * 1.0 + x19*-3.73887877 * 1.0 + x20*-0.49381044 * 1.0 + x22*6.6028795 * 1.0 + x23*-0.5 * 1.0 + x24*3.14159265 * 1.0 + 5.51838556 * 1.0)) .* 0.00330919)  +  (10.1028795 -  (0.10537254 + ((x8*1.0 * 1.0))) ) ) , 3) ) , ...
        ones(length(x1), 1) ];

th = [ -9.37E-6 ; ...
       0.00127044 ; ...
       0.41821565 ; ...
       0.00482128 ; ...
       -0.00288636 ; ...
       3.00754588 ];

V = phi * th;

% MSE = 0.006004248754292726

% complexity = 47


% Configuration:
%         seed: 1
%         nbOfRuns: 5
%         dataset: Datasets_18/8_Left_Ankle_Angle_1000_18.txt
%         maxGenerations: 30000
% Default nbOfThreads: 2
%         epochLength: 1000
%         maxEpochs: 30
%         populationSize: 500
%         nbOfTransformedVar: 40
%         lsIterations: 50
% Default maxNodeEvaluations: 9223372036854775807
%         depthLimit: 5
% Default probHeadNode: 0.1
% Default probTransformedVarNode: 0.2
%         useIdentityNodes: true
% Default saveModelsForPython: false
% Default optMisplacementPenalty: 0.0
% Default desiredOptimum: 
%         regressionClass: LeastSquaresFit
% Default maxWeight: 10000.0
% Default minWeight: 1.0E-20
%         populationClass: PartitionedPopulation
%         resultsDir: Results_18/8_Left_Ankle_Angle/5_1000_30_5_40/
%         tailFunctionSet: Multiply, Plus, Minus, Cosine, Pow2, Pow3, Sine
%         solverName: SolverMultiThreaded
%         nbRegressors: 5
%         nbPredictors: 5
% Default improvementThreshold: 0.0
% Default maxNonImprovingEpochs: 2147483647
% Default identityNodeType: identity