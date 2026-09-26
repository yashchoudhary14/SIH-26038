function [before, after] = tfSamePad(n, k, s)
%TFSAMEPAD TensorFlow "SAME" padding along one axis.
%   Output length is ceil(n/s); any odd remainder goes after (bottom/right).
out = ceil(n / s);
total = max((out - 1) * s + k - n, 0);
before = floor(total / 2);
after = total - before;
end
