function M = cnormc(M)
% Input:  Matrix M
% Output: Matrix M rescaled to have unit norm columns

M = M ./ vecnorm(M,2,1);

end