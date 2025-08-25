function scn = sqcolnorms(F)
% Input:  Matrix F 
% Output: Vector scn containing the squared column norms of F

scn = vecnorm(F,2,1)' .^ 2;

end