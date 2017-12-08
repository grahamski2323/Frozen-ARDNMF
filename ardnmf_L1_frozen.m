function [W, H, lambdas, obj, fit, bound] = ardnmf_L1_frozen(V, beta, tol, n_iter_max, W, H, a, b)

% Majorization-minimization algorithm for ARD-NMF with the beta-divergence
% and L1-norm penalization (exponential prior)
%
%  [W, H, lambdas, obj, fit, bound] = ardnmf_L1(V, beta, tol, n_iter, W_ini, H_ini, a, b)
%
% Input :
%   - V : nonnegative matrix data (F x N)
%   - beta : beta-divergence shape parameter value
%   - tol : tolerance value for convergence
%   - n_iter_max : maximum number of iterations
%   - W : dictionary matrix initialization (F x K)
%   - H : activation matrix initialization (K x N)
%   - a : relevance prior shape parameter 
%   - b : relevance prior scale parameter
%
% We recommend experimenting with several values of 'a', using various
% orders of magnitude. Generally, a good start is a small value compared to
% F+N, say 'a = log(F+N)'. Pruning is increasingly aggressive as the value
% of 'a' decreases. Given a value of 'a', we recommend setting the
% other hyperparameter 'b' to 'sqrt((a-1)*(a-2)*sum(V(:))/(F*K*N))', see
% paper.
%
% Output :
%   - W and H such that
%
%               V \approx W * H
%
%   - lambdas : relevance parameters through iterations
%   - obj : MAP objective through iterations
%   - fit : beta-divergence btw V and WH through iterations
%   - bound : lower bound on relevance paramaters value
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2013 Cedric Fevotte & Vincent Y. F. Tan
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% The reference for this function is
%
% V. Y. F. Tan and C. Févotte. Automatic relevance determination in
% nonnegative matrix factorization with the beta-divergence. IEEE
% Transactions on Pattern Analysis and Machine Intelligence,
% 35(7):1592-1605, July 2013.
%
% Please report any bug at cfevotte -at- unice -dot- fr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[F,N] = size(V);
K = size(W,2);

% To prevent from possible numerical instabilities, the data is added a
% small value (MATLAB's eps) and fitted to WH+eps instead of WH. You may
% set eps = 0 to avoid this but divisions by zero may occur.

% W is signatures, F x K, ASSUME THAT IS CONSTANT
% eps = 0
V = V + eps;
V_ap = W*H + eps;

%cst = F+N+a+1;
% new cst
cst = N+a+1;
bound = b/cst;

scale_W = sum(W,1).'; % FOR USE ONLY IN H UPDATE
scale_H = sum(H,2); 
% inv_lambda = cst./(scale_W+scale_H+b);
inv_lambda = cst./(scale_H+b); % taking out W from regularization term

fit = zeros(1,n_iter_max);
obj = zeros(1,n_iter_max);
lambdas = zeros(K,n_iter_max);

iter = 1;
rel = Inf;
lambdas(:,iter) = 1./inv_lambda;
fit(:,iter) = betadiv(V,V_ap,beta);
%obj(:,iter) = fit(iter) + cst*sum(log(scale_W+scale_H+b));
obj(:,iter) = fit(iter) + cst*sum(log(scale_H+b)); % took out scale_W
fprintf('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E) \n',iter,obj(iter),rel,tol)

while rel > tol
    iter = iter + 1;

    %% Update H %%
    % KEEP THIS THE SAME %
    if  beta > 2
        R = repmat(inv_lambda,1,N);        
        P = W'*(V.*V_ap.^(beta-2));
        Q = W'*V_ap.^(beta-1) + R;
        ex = 1/(beta-1);
    elseif beta == 2 % Euclidean distance
        R = repmat(inv_lambda,1,N);        
        P = W'*V;
        Q = (W'*W)*H + R + repmat(eps*scale_W,1,N); % Use (V_ap*H'+R) if K>F
        ex = 1;        
    elseif (beta < 2) && (beta > 1)
        R = repmat(inv_lambda,1,N);        
        P = W'*(V.*V_ap.^(beta-2));
        Q = W'*V_ap.^(beta-1) + R;
        ex = 1;
    elseif beta == 1 % Generalized KL divergence
        P = W.'*(V./V_ap);
        Q = repmat(scale_W + inv_lambda,1,N);
        ex = 1;        
    elseif beta < 1
        R = repmat(inv_lambda,1,N);        
        P = W'*(V.*V_ap.^(beta-2));
        Q = W'*V_ap.^(beta-1) + R;
        ex = 1/(2-beta);
    end
    
    map = H>0;
    H(map) = H(map).*(P(map)./Q(map)).^ex;
    scale_H = sum(H,2); 
    
    V_ap = W*H + eps;
        
    %% Update W %%
    % skip this, W is frozen %
%     if  beta > 2
%         R = repmat(inv_lambda',F,1);        
%         P = (V.*V_ap.^(beta-2))*H';
%         Q = V_ap.^(beta-1)*H' + R;
%         ex = 1/(beta-1);
%     elseif beta == 2
%         R = repmat(inv_lambda',F,1);        
%         P = V*H';
%         Q = W*(H*H') + R + repmat(eps*scale_H',F,1); % Use (V_ap*H'+R) if K>N
%         ex = 1;        
%     elseif (beta < 2) && (beta > 1)
%         R = repmat(inv_lambda',F,1);        
%         P = (V.*V_ap.^(beta-2))*H';
%         Q = V_ap.^(beta-1)*H' + R;
%         ex = 1;
%     elseif beta == 1
%         P = (V./V_ap)*H.';
%         Q = repmat(scale_H'+inv_lambda',F,1);
%         ex = 1;
%     elseif beta < 1
%         R = repmat(inv_lambda',F,1);        
%         P = (V.*V_ap.^(beta-2))*H';
%         Q = V_ap.^(beta-1)*H' + R;
%         ex = 1/(2-beta);                
%     end
%     
%     map = W>0;
%     W(map) = W(map).*(P(map)./Q(map)).^ex;
%     scale_W = sum(W,1).'; 
%     
%     V_ap = W*H + eps;
    
    %% Update lambda %%
%     inv_lambda = cst./(scale_W+scale_H+b);
    inv_lambda = cst./(scale_H+b);
    
    %% Monitor %%
    fit(iter) = betadiv(V,V_ap,beta);
%     obj(iter) = fit(iter) + cst*sum(log(scale_W+scale_H+b));
    obj(iter) = fit(iter) + cst*sum(log(scale_H+b));
    lambdas(:,iter) = 1./inv_lambda;
    
    % Compute relative change of the relevance parameters
    rel = max(abs((lambdas(:,iter)-lambdas(:,iter-1))./lambdas(:,iter)));
    
    % Display objective value and relative change every 500 iterations
    if rem(iter,500)==0
        fprintf('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E) \n',iter,obj(iter),rel,tol)        
    end
    
end

% Trim variables
fit = fit(1:iter);
obj = obj(1:iter); 
lambdas = lambdas(:,1:iter);
% Add constant to optain true minus log posterior value
obj = obj + (K*cst*(1-log(cst))); 
% Display final values
fprintf('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E) \n',iter,obj(iter),rel,tol)
if iter == n_iter_max
    fprintf('Maximum number of iterations reached (n_iter_max = %d) \n',n_iter_max)
end