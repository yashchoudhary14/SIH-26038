function Wm = resizeWeights(n, N, method)
%RESIZEWEIGHTS Sparse N-by-n matrix mapping an axis of length n to length N,
%   reproducing OpenCV's per-method coordinate mapping and border handling.

% OpenCV forms the source step as 1/(dst/src), not src/dst; the two can differ
% in the last bit, which moves floor(x * scale) across an integer (it changed
% the nearest-neighbour FOV mask of one real photograph).
scale = 1 / (N / n);
rows = [];
cols = [];
vals = [];

switch lower(method)
    case 'nearest'
        % INTER_NEAREST: sx = floor(dx * src/dst), clamped.
        s = min(floor((0:N-1) * scale), n - 1);
        rows = 1:N; cols = s + 1; vals = ones(1, N);

    case 'linear'
        fx = ((0:N-1) + 0.5) * scale - 0.5;
        sx = floor(fx);
        f = fx - sx;
        lo = sx < 0;          f(lo) = 0; sx(lo) = 0;
        hi = sx >= n - 1;     f(hi) = 0; sx(hi) = n - 1;
        s2 = min(sx + 1, n - 1);
        rows = [1:N, 1:N];
        cols = [sx + 1, s2 + 1];
        vals = [1 - f, f];

    case 'cubic'
        % INTER_CUBIC, Keys kernel with A = -0.75; out-of-range taps fold onto
        % the edge pixel (OpenCV clamps the index, i.e. replicate).
        A = -0.75;
        fx = ((0:N-1) + 0.5) * scale - 0.5;
        sx = floor(fx);
        f = fx - sx;
        w0 = ((A * (f + 1) - 5 * A) .* (f + 1) + 8 * A) .* (f + 1) - 4 * A;
        w1 = ((A + 2) * f - (A + 3)) .* f .* f + 1;
        w2 = ((A + 2) * (1 - f) - (A + 3)) .* (1 - f) .* (1 - f) + 1;
        w3 = 1 - w0 - w1 - w2;
        taps = {w0, w1, w2, w3};
        for k = 0:3
            idx = min(max(sx + k - 1, 0), n - 1);
            rows = [rows, 1:N];              %#ok<AGROW>
            cols = [cols, idx + 1];          %#ok<AGROW>
            vals = [vals, taps{k + 1}];      %#ok<AGROW>
        end

    case 'area'
        if n >= N
            % Downscale: exact fractional-overlap weights, as
            % computeResizeAreaTab builds them.
            for d = 0:N-1
                fsx1 = d * scale;
                fsx2 = fsx1 + scale;
                cellW = min(scale, n - fsx1);
                sx1 = ceil(fsx1);
                sx2 = floor(fsx2);
                sx2 = min(sx2, n - 1);
                sx1 = min(sx1, sx2);
                if sx1 - fsx1 > 1e-3
                    rows(end+1) = d + 1; cols(end+1) = sx1; %#ok<AGROW>
                    vals(end+1) = (sx1 - fsx1) / cellW;     %#ok<AGROW>
                end
                for sx = sx1:sx2-1
                    rows(end+1) = d + 1; cols(end+1) = sx + 1; %#ok<AGROW>
                    vals(end+1) = 1 / cellW;                   %#ok<AGROW>
                end
                if fsx2 - sx2 > 1e-3
                    rows(end+1) = d + 1; cols(end+1) = sx2 + 1; %#ok<AGROW>
                    vals(end+1) = min(min(fsx2 - sx2, 1), cellW) / cellW; %#ok<AGROW>
                end
            end
        else
            % Upscale with INTER_AREA degrades to OpenCV's area-aware linear.
            inv = N / n;
            sx = floor((0:N-1) * scale);
            f = ((0:N-1) + 1) - (sx + 1) * inv;
            f(f <= 0) = 0;
            f = f - floor(f);
            s1 = min(sx, n - 1);
            s2 = min(sx + 1, n - 1);
            rows = [1:N, 1:N];
            cols = [s1 + 1, s2 + 1];
            vals = [1 - f, f];
        end

    otherwise
        error('drscreen:cv:resize', 'Unknown resize method "%s".', method);
end

Wm = sparse(rows, cols, vals, N, n);
end
