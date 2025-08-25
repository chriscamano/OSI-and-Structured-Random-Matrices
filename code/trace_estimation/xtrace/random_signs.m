function Om = random_signs(n,s)
% Input:  Dimensions n, s for random sign matrix
% Output: An n*s random sign matrix Om

    Om = -3 + 2*randi(2,n,s);

end