function trace_est = na_hutchplusplus_KR(matVecOracle, num_queries, dimension, ell, args)
    % set defaults
    if ~isfield(args,'hutch_dist'),  args.hutch_dist  = @(m,n) 2*randi(2,m,n)-3; end
    if ~isfield(args,'sketch_dist'), args.sketch_dist = @(m,n) 2*randi(2,m,n)-3; end
    if ~isfield(args,'c1'),          args.c1          = 1/6;               end
    if ~isfield(args,'c2'),          args.c2          = 1/3;               end

    % wrap A into oracle if needed
    if ~isa(matVecOracle,'function_handle')
        dimension    = size(matVecOracle,1);
        A            = matVecOracle;
        matVecOracle = @(X) A * X;
    elseif isempty(dimension) || dimension<1
        error('Must supply ''dimension'' when passing a handle.');
    end

    Rq = round(num_queries * args.c1);
    Sq = round(num_queries * args.c2);
    Gq = num_queries - Rq - Sq;

    R = krSketch(dimension, Rq, ell, 'gaussian', true);
    S = krSketch(dimension, Sq, ell, 'gaussian', true);
    G = krSketch(dimension, Gq, ell, 'gaussian', true);

    Z = matVecOracle(R);   % A*R
    W = matVecOracle(S);   % A*S
    H = matVecOracle(G);   % A*G

    SZ = S' * Z;      % size Sq × Rq
    WZ = W' * Z;      % size Sq × Rq
    GZ = G' * Z;      % size Gq × Rq
    WG = W' * G;      % size Sq × Gq
    
    [Q_s, R_s] = qr(SZ, 0); 
    X = R_s \ (Q_s' * WZ); 
    T1 = trace(X);
    Y = R_s \ (Q_s' * WG);
    Hterm = trace(G' * H);
    Corr  = trace(GZ * Y);
    T2    = (1/Gq) * (Hterm - Corr);
    trace_est = T1 + T2;
end