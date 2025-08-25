function est = hutch_norm_est(A, Q)
    trials = 10;
    Ome   = randn(size(Q,1), trials);
    Y     = A(Ome);
    R     = Y - Q * (Q' * Y);
    est   = norm(R, 'fro') / sqrt(trials);
end