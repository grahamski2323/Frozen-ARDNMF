function d = betadiv(A,B,beta)

% Computes beta-divergence between matrices
%
% betadiv(A,B,b) = = \sum \sum d(a_ij,b_ij)
%
% with
% 
% - beta \= 0,1
%   d(x|y) = (x^beta + (beta-1)*y^beta - beta*x*y^(beta-1))/(beta*(beta-1))
%
% - beta = 1 (Generalized Kullback-Leibler divergence)
%   d(x|y) = x*log(x/y) - x + y
%
% - beta = 0 (Itakura-Saito divergence)
%   d(x|y) = x/y - log(x/y) - 1

switch beta
    case 2
        d = sum((A(:)-B(:)).^2)/2;
    case 1
        ind_0 = find(A(:)<=eps);
        ind_1 = 1:length(A(:));
        ind_1(ind_0) = [];
        d = sum( A(ind_1).*log(A(ind_1)./B(ind_1)) - A(ind_1) + B(ind_1) ) + sum(B(ind_0));        
    case 0
        d = sum( A(:)./B(:) - log(A(:)./B(:)) ) - length(A(:));
    otherwise
        d = sum( A(:).^beta + (beta-1)*B(:).^beta - beta*A(:).*B(:).^(beta-1) )/(beta*(beta-1));
end

