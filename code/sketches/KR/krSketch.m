% Copyright (c) 2025, Chris Camano

function Om = krSketch(n, k, l, dist, sphere)
    % krSketch   Build an n×k Khatri–Rao sketch with five distribution options
    %
    %    Om = krSketch(n, k, l)
    %    Om = krSketch(n, k, l, dist, sphere)
    %
    %  Inputs
    %    n      — target number of rows
    %    k      — target number of columns
    %    l      — number of factors in the KR product (so OmFull is m^l×k)
    %    dist   — (optional) sampling distribution:
    %             'gaussian'     (default)
    %             'cgaussian'    (complex Gaussian)
    %             'rademacher'   (±1)
    %             'crademacher'  (±1±1i)
    %             'steinhaus'    (uniform on the complex unit sphere)
    %    sphere — (optional, default=false) if true, normalize each column
    %             of the final n×k matrix to unit ℓ₂-norm (uniform on Sⁿ⁻¹)
    %
    %  Output
    %    Om     — an n×k sketch matrix
    
    if nargin < 4, dist   = 'gaussian'; end
    if nargin < 5, sphere = false;      end

    d0 = ceil(n^(1/l));
    Oms = cell(1, l);

    for j = 1:l
        switch lower(dist)
            case 'gaussian'
                B = randn(d0, k);
            case 'cgaussian'
                B = (randn(d0, k) + 1i*randn(d0, k)) / sqrt(2);
            case 'rademacher'
                B = sign(rand(d0, k) - 0.5);
            case 'crademacher'
                B = (sign(rand(d0, k) - 0.5) ...
                   + 1i*sign(rand(d0, k) - 0.5)) / sqrt(2);
            case 'steinhaus'
                B = exp(1i * 2*pi * rand(d0, k));
            otherwise
                error('krSketch:unknownDist', 'Unrecognized dist ''%s''.', dist);
        end

        if sphere
            B = B ./ vecnorm(B, 2, 1);
            B = B * sqrt(d0);
        end

        Oms{j} = B;
    end
    OmFull = kr(Oms);
    Om     = OmFull(1:n, :);
    Om = Om ./ sqrt(k);
end