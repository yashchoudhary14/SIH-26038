function D = distanceTransform(bw)
%DISTANCETRANSFORM cv2.distanceTransform(bw, DIST_L2, 5).
%
%   OpenCV's 5x5 mask is a two-pass chamfer approximation with weights
%   (1, 1.4, 2.1969) in 16-bit fixed point -- not the exact Euclidean distance
%   bwdist returns. The venous-beading cue compares each vessel segment's
%   calibre against thresholds (1.8 px, ratio 1.9) that were set on this
%   chamfer metric, so the chamfer is reproduced here.

HV = 65536; DIAG = 91750; LONG = 143976;   % cvRound(w * 2^16)
INF = 2147483647;
bw = bw ~= 0;
[h, w] = size(bw);
B = 2;
T = INF * ones(h + 2 * B, w + 2 * B);

% forward pass
for i = 1:h
    ii = i + B;
    for j = 1:w
        jj = j + B;
        if ~bw(i, j)
            T(ii, jj) = 0;
        else
            t0 = min([T(ii-2, jj-1) + LONG, T(ii-2, jj+1) + LONG, ...
                      T(ii-1, jj-2) + LONG, T(ii-1, jj-1) + DIAG, ...
                      T(ii-1, jj)   + HV,   T(ii-1, jj+1) + DIAG, ...
                      T(ii-1, jj+2) + LONG, T(ii, jj-1)   + HV]);
            T(ii, jj) = t0;
        end
    end
end

% backward pass
D = zeros(h, w, 'single');
for i = h:-1:1
    ii = i + B;
    for j = w:-1:1
        jj = j + B;
        t0 = T(ii, jj);
        if t0 > HV
            t0 = min([t0, T(ii+2, jj+1) + LONG, T(ii+2, jj-1) + LONG, ...
                      T(ii+1, jj+2) + LONG, T(ii+1, jj+1) + DIAG, ...
                      T(ii+1, jj)   + HV,   T(ii+1, jj-1) + DIAG, ...
                      T(ii+1, jj-2) + LONG, T(ii, jj+1)   + HV]);
            T(ii, jj) = t0;
        end
        D(i, j) = single(t0 / 65536);
    end
end
end
