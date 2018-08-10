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

phi = [ ( ( power(((x1*-0.5 * 1.0 + x2*11.11111111 * 1.97863696 + x3*-4.89055346 * 1.0 + x4*-1.71 * 1.0 + x5*2.23606798 * 1.0 + x7*2.32509607 * 1.0 + x8*-2.32509607 * 1.28535444 + x9*-2.32509607 * 1.19749356 + x10*5.0 * 1.05232904 + x11*16.0 * 1.0 + x12*-0.56649658 * 1.0 + x13*3.14159265 * 1.0 + x14*0.40824829 * 1.18353547 + x15*0.25 * 1.0 + x16*-0.20412415 * 1.0 + x17*-0.62831853 * 1.0 + x18*-2.0 * 1.0 + x19*-0.0625 * 1.0 + x20*-1.0 * 1.0 + x21*-0.14159265 * 1.0 + x22*-0.25 * 1.0 + x23*1.14159265 * 1.0 + x24*1.0 * 1.0 + 4.0 * 1.0)), 2)  -  (((x1*0.5 * 1.0 + x2*-2.0 * 0.48016597 + x3*-1.71 * 1.05007277 + x5*8.0 * 0.94014521 + x7*4.39244837 * 0.24944842 + x9*-1.42823378 * 0.04909871 + x10*8.05849823 * 1.49747228 + x11*-17.42591641 * 2.38967226 + x12*0.15196635 * 1.70280982 + x13*3.33333333 * 1.0 + x14*-1.71 * 0.97527407 + x15*-1.75 * 1.19774481 + x16*-0.5 * 1.0 + x17*-1.42823378 * 0.86131559 + x18*0.42087585 * 1.0 + x19*-5.37287753 * 0.30285815 + x20*-0.20412415 * 1.0920725 + x21*1.30323378 * 1.10037417 + x22*0.5 * 1.0 + x23*0.66666667 * 1.0 + x24*-0.31830989 * 1.0 + 6.2831853 * 1.0)) + ((x23*1.0 * 1.0))) ) ) , ...
        ( ( power(((x13*1.0 * 1.0)), 2)  -  power(((x1*-0.5 * 1.0 + x2*11.11111111 * 1.97863696 + x3*-4.89055346 * 1.0 + x4*-1.71 * 1.0 + x5*2.23606798 * 1.0 + x7*2.32509607 * 1.0 + x8*-2.32509607 * 1.28535444 + x9*-2.32509607 * 1.19749356 + x10*5.0 * 1.05232904 + x11*16.0 * 1.0 + x12*-0.56649658 * 1.0 + x13*3.14159265 * 1.0 + x14*0.40824829 * 1.18353547 + x15*0.25 * 1.0 + x16*-0.20412415 * 1.0 + x17*-0.62831853 * 1.0 + x18*-2.0 * 1.0 + x19*-0.0625 * 1.0 + x20*-1.0 * 1.0 + x21*-0.14159265 * 1.0 + x22*-0.25 * 1.0 + x23*1.14159265 * 1.0 + x24*1.0 * 1.0 + 4.0 * 1.0)), 2) ) ) , ...
        ( (((x2*-0.31830989 * 1.0 + x3*8.05849823 * 1.0 + x4*-2.32509607 * 1.0 + x5*8.05849823 * 1.0 + x6*-0.5 * 3.39820811 + x7*-0.125 * 0.76274608 + x8*-1.68350342 * 1.342036 + x9*6.2831853 * 1.40169894 + x10*16.0 * 1.0 + x11*0.15196635 * 1.0 + x12*-0.12831853 * 1.0 + x13*-0.81649658 * 1.0 + x14*-1.71 * 1.0 + x15*1.30323378 * 1.0 + x16*0.5 * 1.0 + x18*-0.25 * 0.477379 + x19*0.25 * 1.0 + x20*0.2 * 1.0 + x21*-0.56649658 * 1.0 + x23*0.5 * 1.0 + x24*0.25 * 1.0 + 0.25 * 1.0)) .*  cos(((x9*1.0 * 1.0))) ) ) , ...
        (((x11*1.0 * 1.0))) , ...
        ( ( (0.54433105 +  (-0.14159265 .* ((x2*1.0 * 1.0))) )  -  ( cos(((x1*1.0 * 1.0 + x2*-2.0 * 1.0 + x4*-0.125 * 1.0 + x5*8.05849823 * 1.0 + x6*0.33333333 * 1.0 + x7*0.78539816 * 0.69584178 + x8*0.78539816 * 1.0 + x9*0.33333333 * 1.66766525 + x10*1.52482657 * 1.0 + x11*1.57079633 * 1.0 + x12*4.39244837 * 0.41521761 + x14*-2.0 * 1.0 + x15*2.32509607 * 1.0 + x16*-0.0625 * 1.0 + x19*0.0625 * 1.09071298 + x20*-1.18350342 * 1.38870899 + x22*1.57079633 * 1.0 + x23*3.14159265 * 1.06335466 + x24*-0.2 * 1.0 + 15.66666667 * 1.0)))  .* 6.4E-5) ) ) , ...
        ones(length(x1), 1) ];

th = [ -4.295E-5 ; ...
       -4.734E-5 ; ...
       -1.7793E-4 ; ...
       0.02446629 ; ...
       -6.81843244 ; ...
       3.7124079 ];

V = phi * th;

% MSE = 5.159283600927296E-6

% complexity = 42


% Configuration:
%         seed: 1
%         nbOfRuns: 5
%         dataset: Datasets_18/2_Torso_zpos_1000_18.txt
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
%         resultsDir: Results_18/2_Torso_zpos/5_1000_30_5_40/
%         tailFunctionSet: Multiply, Plus, Minus, Cosine, Pow2, Pow3, Sine
%         solverName: SolverMultiThreaded
%         nbRegressors: 5
%         nbPredictors: 5
% Default improvementThreshold: 0.0
% Default maxNonImprovingEpochs: 2147483647
% Default identityNodeType: identity
