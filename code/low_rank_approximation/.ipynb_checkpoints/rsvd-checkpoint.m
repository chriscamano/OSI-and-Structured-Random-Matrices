function [U, S, V, elapsed, relErr] = rsvd(A, Omega, recordTime)
    %RSVD   Compute rank-k approximation via HMT randomized SVD.
    %
    % Inputs
    %   A          — (n×m) data matrix
    %   Omega      — (m×k) sketching matrix
    %   recordTime — logical flag; if true, measure time and compute error
    %
    % Outputs
    %   U       — (n×k) left singular vectors
    %   S       — (k×k) diagonal matrix of singular values
    %   V       — (m×k) right singular vectors
    %   elapsed — time for sketch/QR/SVD steps (empty if recordTime=false)
    %   relErr  — ‖A − U*S*V'‖_F / ‖A‖_F (empty if recordTime=false)

    if nargin < 3
        recordTime = false;                                     
    end
    if recordTime
        t0 = tic;                                              
    end

    %===========================================================
    Y = A * Omega;                                             % Sketch
    [Q, ~] = qr(Y, 0);                                         % Orthonormalize
    B = Q' * A;                                          
    [UU, S, V] = svd(B, 'econ');                               % Economy-size SVD
    U = Q * UU;                                                % Project
    %===========================================================

    if recordTime
        elapsed = toc(t0);                                    
        Ahat    = U * S * V';                                  % approximate reconstruction
        relErr  = norm(A - Ahat, 'fro') / norm(A, 'fro');      % relative Frobenius error
    else
        elapsed = [];                                         
        relErr  = [];
    end
end



