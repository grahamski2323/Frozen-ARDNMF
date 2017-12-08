% Demo file for ARD-NMF
% Just run demo.m in Matlab

%% Load data
tic();
mutation_dat = importdata('complete_mutation_matrix_genome.txt');
mut_cat = mutation_dat.textdata(1,2:end); 
samples = mutation_dat.textdata(2:end,1);
% for simplicity and ease, M = V, input mutation matrix
V = mutation_dat.data;
V = V'; % to match the matrix dimensions of this algorithm
[F,N] = size(V);

%% ALPHABETIZING to match with COSMIC
% note, will have to re-order this for appropraite plotting of signatures
% like in the papers
[mut_cat_V,id] = sort(mut_cat);
V = V(id,:); % rearrangingin alphabetical order

%% Set parameters
beta = 1; % Beta-divergence shape parameter
K = 30; % for 30 COSMIC signatures
a = 10; % Relevance parameters shape parameter
tol = 1e-8; % Tolerance value for convergence
n_iter_max = 10000; % Max nb of iterations
L = 1; % L1-ARD (L=1) or L2-ARD (L = 2)

%% Initialize
 mean_V = sum(V(:))/(F*N); % Data sample mean per component
%W_ini = (rand(F,K) + 1)*(sqrt(mean_V/K));
H_ini = (rand(K,N) + 1)*(sqrt(mean_V/K));

% new W, with frozen
W_frz = importdata('signatures_probabilities.txt');
W_ini = W_frz.data;
sig_labels = W_frz.textdata(1,4:end);
mut_cat_cosmic = W_frz.textdata(2:end,3);
clear W_frz
clear mutation_dat

%% rearranging the mutation cateogries for plotting purposes
test = {};
tt = [
'A[C>A]A'
'A[C>A]C'
'A[C>A]G'
'A[C>A]T'
'C[C>A]A'
'C[C>A]C'
'C[C>A]G'
'C[C>A]T'
'G[C>A]A'
'G[C>A]C'
'G[C>A]G'
'G[C>A]T'
'T[C>A]A'
'T[C>A]C'
'T[C>A]G'
'T[C>A]T'
'A[C>G]A'
'A[C>G]C'
'A[C>G]G'
'A[C>G]T'
'C[C>G]A'
'C[C>G]C'
'C[C>G]G'
'C[C>G]T'
'G[C>G]A'
'G[C>G]C'
'G[C>G]G'
'G[C>G]T'
'T[C>G]A'
'T[C>G]C'
'T[C>G]G'
'T[C>G]T'
'A[C>T]A'
'A[C>T]C'
'A[C>T]G'
'A[C>T]T'
'C[C>T]A'
'C[C>T]C'
'C[C>T]G'
'C[C>T]T'
'G[C>T]A'
'G[C>T]C'
'G[C>T]G'
'G[C>T]T'
'T[C>T]A'
'T[C>T]C'
'T[C>T]G'
'T[C>T]T'
'A[T>A]A'
'A[T>A]C'
'A[T>A]G'
'A[T>A]T'
'C[T>A]A'
'C[T>A]C'
'C[T>A]G'
'C[T>A]T'
'G[T>A]A'
'G[T>A]C'
'G[T>A]G'
'G[T>A]T'
'T[T>A]A'
'T[T>A]C'
'T[T>A]G'
'T[T>A]T'
'A[T>C]A'
'A[T>C]C'
'A[T>C]G'
'A[T>C]T'
'C[T>C]A'
'C[T>C]C'
'C[T>C]G'
'C[T>C]T'
'G[T>C]A'
'G[T>C]C'
'G[T>C]G'
'G[T>C]T'
'T[T>C]A'
'T[T>C]C'
'T[T>C]G'
'T[T>C]T'
'A[T>G]A'
'A[T>G]C'
'A[T>G]G'
'A[T>G]T'
'C[T>G]A'
'C[T>G]C'
'C[T>G]G'
'C[T>G]T'
'G[T>G]A'
'G[T>G]C'
'G[T>G]G'
'G[T>G]T'
'T[T>G]A'
'T[T>G]C'
'T[T>G]G'
'T[T>G]T'
];
for i = 1:size(tt,1)
    test{i} = tt(i,:);
end
test = test';
[tf,idx] = ismember(test',mut_cat_cosmic');
mut_cat_cosmic = mut_cat_cosmic(idx);
W_ini = W_ini(idx,:);
V = V(idx,:);
%% Display a few random data samples
figure;

for i = 1:3
    subplot(3,1,i);
    n = randi(N);
    bar(V(:,n));
    title([samples(n)])
end
%% plotting the COSMIC signatures
figure;
for k=1:K
    subplot(ceil(K/3),3,k);
    bar(W_ini(:,k));
    ylim([0 0.2])
    title(['Signature ' num2str(k)])    
end

%% Run the algorithm
if L == 1 % L1-ARD
    
    % Set b using method of moments (see paper)
    b = sqrt((a-1)*(a-2)*mean_V/K);
    [W, H, lambdas, obj, fit, bound] = ardnmf_L1_frozen(V, beta, tol, n_iter_max, W_ini, H_ini, a, b);
    
elseif L == 2 % L2-ARD
    
    % Set b using method of moments (see paper)
    b = (pi/2)*(a-1)*mean_V/K;
    [W, H, lambdas, obj, fit, bound] = ardnmf_L2_frozen(V, beta, tol, n_iter_max, W_ini, H_ini, a, b);
    
end

%% Display fit and relevance parameters
figure;

if sum(obj<=0) == 0
    subplot(411);
    loglog(obj,'k');
    axis tight;
    title('objective function')
end

subplot(412);
loglog(fit,'k');
axis tight;
title('fit to data (beta-divergence)')

subplot(212)
plot(lambdas'-bound,'k')
xlim([1 length(fit)])
title('relevance')

%% Display learnt signatures

% Sort variables according to relevance
[junk,order] = sort(lambdas(:,end),1,'descend');
W_o = W(:,order);
H_o = H(order,:);
lambda_o = lambdas(order,end);

% Rescale W by prior expectation for improved readability (see paper)
% ONLY TO SHOW WHICH ONES ARE ZEROED OUT
if L == 1
    W_sc = W_o * diag(lambda_o);
elseif L == 2
    W_sc = W_o * diag(sqrt(2*lambda_o/pi));
end

% Display
figure;
% all signatures, scaling them by relevance
% pruned out signatures should have y-axis of about 1e-4
for k=1:K
    subplot(ceil(K/3),3,k);
    bar(W_sc(:,k))
    title(['Signature ' num2str(order(k))])    
end
%figure;

% % unscaled plot, to show relevance order of sigs
% for k=1:K
%     subplot(ceil(K/3),3,k);
%     bar(W_o(:,k))
%     ylim([0 0.2]);
%     title(['Signature ' num2str(order(k))])    
% end

% figuring out Keff
lamthresh = (lambda_o - bound)./(bound);
tol = 5; % this threshold can be any value, even 10e-8 
remove = find(lamthresh <= tol);

% only keeping relevant signatures
W_o(:,remove) = [];
H_o(remove,:) = [];
lambda_o(remove) = [];
W_eff = W_o;
H_eff = H_o;
lambda_eff = lambda_o;
K_eff = length(lambda_eff);
disp(['Number of Signatures Determined: ' num2str(K_eff)]);
endtime = toc();
disp(['Total Run Time: ' num2str(endtime)]);

%showing unscaled, relevant signautre output
figure;
for k=1:K_eff
    subplot(ceil(K_eff/2),2,k);
    bar(W_eff(:,k));
    ylim([0 0.2]);
    title(['Signature ' num2str(order(k))])
end

