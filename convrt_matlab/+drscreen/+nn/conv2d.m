function Y = conv2d(X, W, b, stride)
%CONV2D 2-D cross-correlation with TensorFlow "SAME" padding.
%
%   Y = drscreen.nn.conv2d(X, W, b, stride)
%
%   X : H x W x C single (or gpuArray).
%   W : [kh kw C O]         dense convolution, or
%       [kh kw 1 1 C]       depthwise (one filter per channel).
%   b : [] or a vector of O (dense) / C (depthwise) biases.
%
%   Dense convolution is kh*kw shifted matrix products -- plain MATLAB, no
%   Deep Learning Toolbox. Padding is TF "SAME" (extra on the bottom/right
%   when the total is odd), which is what timm's Conv2dSame does for the
%   stride-2 layers and what the symmetric (k-1)/2 padding of the stride-1
%   layers amounts to. PyTorch convolution is cross-correlation, and so is this.
%
%   If the Deep Learning Toolbox happens to be installed, dlconv does the same
%   computation 3-7x faster (see drscreen.nn.useToolbox); the weight layouts
%   above are dlconv's own, so nothing is rearranged.

if nargin < 3, b = []; end
if nargin < 4 || isempty(stride), stride = 1; end
[H, Wd, C] = size(X);
kh = size(W, 1);
kw = size(W, 2);
[pt, pb] = drscreen.nn.tfSamePad(H, kh, stride);
[pl, pr] = drscreen.nn.tfSamePad(Wd, kw, stride);
Ho = ceil(H / stride);
Wo = ceil(Wd / stride);

if drscreen.nn.useToolbox()
    Y = extractdata(dlconv(dlarray(X, 'SSC'), W, 0, 'Stride', stride, ...
                           'Padding', [pt pl; pb pr]));
    if ~isempty(b)
        Y = Y + reshape(b, 1, 1, []);
    end
    return
end

if pt + pb + pl + pr > 0
    Xp = zeros(H + pt + pb, Wd + pl + pr, C, 'like', X);
    Xp(pt + 1:pt + H, pl + 1:pl + Wd, :) = X;
else
    Xp = X;
end
rs = @(u) u:stride:u + stride * (Ho - 1);
cs = @(v) v:stride:v + stride * (Wo - 1);

depthwise = size(W, 5) > 1;
if depthwise
    Y = zeros(Ho, Wo, C, 'like', X);
    for u = 1:kh
        for v = 1:kw
            Y = Y + Xp(rs(u), cs(v), :) .* reshape(W(u, v, 1, 1, :), 1, 1, C);
        end
    end
else
    O = size(W, 4);
    if kh == 1 && kw == 1 && stride == 1
        Y = reshape(X, H * Wd, C) * reshape(W, C, O);
    else
        Y = zeros(Ho * Wo, O, 'like', X);
        for u = 1:kh
            for v = 1:kw
                S = Xp(rs(u), cs(v), :);
                Y = Y + reshape(S, Ho * Wo, C) * reshape(W(u, v, :, :), C, O);
            end
        end
    end
    Y = reshape(Y, Ho, Wo, O);
end

if ~isempty(b)
    Y = Y + reshape(b, 1, 1, []);
end
end
