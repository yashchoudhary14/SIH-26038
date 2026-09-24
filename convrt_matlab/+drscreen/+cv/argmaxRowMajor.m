function [r, c, v] = argmaxRowMajor(A)
%ARGMAXROWMAJOR First maximum in row-major order, as numpy.argmax finds it.
%
%   MATLAB's max scans column-major, so on a tie it returns a different pixel.
%   Returns 0-based (row, col) and the value.

v = max(A(:));
[rr, cc] = find(A == v);
k = find(rr == min(rr));
[~, j] = min(cc(k));
r = rr(k(j)) - 1;
c = cc(k(j)) - 1;
end
