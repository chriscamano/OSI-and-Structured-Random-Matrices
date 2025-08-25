function srn = sqrownorms(F)
% Input:  Matrix F 
% Output: Vector srn containing the squared row norms of F

srn = vecnorm(F,2,2) .^ 2;

end