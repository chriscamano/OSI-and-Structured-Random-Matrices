function Snap = GPEsim(Nx, L, g, Omega_max, dtau, maxIter, tol, saveEvery)
% GPEsim  Run imaginary‐time evolution of the GPE with ramped stirring.
%
%   Snap = GPEsim(Nx, L, g, Omega_max, dtau, maxIter, tol, saveEvery)
%
% Inputs:
%   Nx         – grid points per dimension
%   L          – half‐width of domain
%   g          – interaction strength (β₂)
%   Omega_max  – maximal rotation speed
%   dtau       – imaginary‐time step size
%   maxIter    – maximum number of iterations
%   tol        – convergence tolerance
%   saveEvery  – save snapshot every saveEvery steps
%
% Output:
%   Snap       – (Nx^2 × nSnap) matrix of complex snapshots

    dx      = 2*L/(Nx-1);
    x       = linspace(-L, L, Nx);
    [X, Y]  = meshgrid(x, x);

    fprintf('--- GPE Imag‐Time Snapshots + POD ---\n');
    fprintf('Nx=%d, L=%.1f, dx=%.3f, g=%.0f\n', Nx, L, dx, g);
    fprintf('Imag‐time: dtau=%.1e, maxIter=%d, tol=%.1e, saveEvery=%d\n\n', ...
            dtau, maxIter, tol, saveEvery);

    % Gaussian seed 
    R2    = X.^2 + Y.^2;
    s     = 1;  
    phi0  = exp(-0.5 * R2 / s^2);
    a     = 0.05;
    phi0  = phi0 .* exp(1i * a * randn(size(X))); %phase defect
    phi   = phi0 / sqrt(sum(abs(phi0(:)).^2) * dx^2);

    nSnapEst = ceil(maxIter / saveEvery);
    Snap     = complex(zeros(Nx^2, nSnapEst));
    snapIdx  = 0;

    % pre compute GPE ops
    dk      = pi / L;
    kidx    = ifftshift(-floor(Nx/2):(ceil(Nx/2)-1));
    kvec    = dk * kidx;
    [KX, KY]= meshgrid(kvec, kvec);
    LapSym  = -(KX.^2 + KY.^2);
    X2      = X.^2;
    Y2      = Y.^2;

    % anisotropy parameters (off)
    gamma_i = 1;
    gamma_f = 1.0;

    rampStep = round(0.2 * maxIter);

    for it = 1:maxIter
        phi_old = phi;

        ph_hat = fft2(phi);
        lapPhi = ifft2(LapSym .* ph_hat);
        dpx    = ifft2(1i * KX .* ph_hat);
        dpy    = ifft2(1i * KY .* ph_hat);
        LzPhi  = -1i * (X .* dpy - Y .* dpx);

        % interpolate anisotropy
        frac   = max(0, (maxIter - it)/maxIter);
        gamma  = gamma_f + frac*(gamma_i - gamma_f);
        Vtrap  = 0.5 * (X2 + (gamma * Y).^2);

        % ramp up stirring
        if it <= rampStep
            Omega = 0;
        else
            Omega = Omega_max;
        end

        rhs = 0.5*lapPhi ...
            - Vtrap .* phi ...
            - g * abs(phi).^2 .* phi ...
            + Omega * LzPhi;

        phi = phi + dtau * rhs;
        phi = phi / sqrt(sum(abs(phi(:)).^2) * dx^2);

        if mod(it, saveEvery) == 0
            snapIdx          = snapIdx + 1;
            Snap(:, snapIdx) = phi(:);
        end

        if mod(it, 500) == 0
            dmax = max(abs(phi(:) - phi_old(:)));
            fprintf('  iter %5d/%5d, Δ=%.2e, Omega=%.2f\n', ...
                    it, maxIter, dmax, Omega);
        end
    end

    Snap = Snap(:,1:snapIdx);
end