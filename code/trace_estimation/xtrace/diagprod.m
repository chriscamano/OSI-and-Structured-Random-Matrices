function d = diagprod(F,G)
% Input:  Matrices F and G of the same size
% Output: The diagonal of the product d = diag(G'*F)

d = sum(conj(F).*G,1).';

end