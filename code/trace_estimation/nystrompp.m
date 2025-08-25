function trest = nystrompp(n,Afun,Omega,Psi,matvecs)
%Output
%   trest: Estimate of trace(A)
%   approximation_error: sqrt(norm(A,'fro')^2-approximation_error) gives
%   approximation error in Frobenius norm
%   cond_number: Condition number of middle factor in Nyström approximation

%Compute matrix products with A
Y = Afun(Omega); 
Z = Afun(Psi);

%Regularizaton
nu = sqrt(n)*eps(norm(Y));
Y_nu = Y + nu*Omega;

C = chol(real(Omega'*Y_nu));
B = Y_nu/C;
[U,S,~] = svd(B,'econ');
Lambda = max(0,S^2-nu*eye(size(S)));

tr1 = trace(Lambda);
tr2 = 2*(product_trace(Psi',Z) - product_trace(Psi',U*(Lambda*(U'*Psi))))/matvecs;
trest = tr1 + tr2;

end

function t = product_trace(A,B)
t = sum(sum(A.*B',2));
end