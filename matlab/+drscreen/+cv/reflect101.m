function idx = reflect101(n, lo, hi)
%REFLECT101 1-based source indices for 0-based positions lo..hi.
%
%   OpenCV's default border, BORDER_REFLECT_101: "gfedcb|abcdefgh|gfedcba" --
%   the edge pixel is not repeated. MATLAB's padarray 'symmetric' repeats it,
%   which shifts every filtered value near the frame edge, so filters here pad
%   by index instead. Handles pads wider than the signal (multiple reflections),
%   which the large illumination kernels do produce.

p = lo:hi;
if n == 1
    idx = ones(size(p));
    return
end
period = 2 * (n - 1);
p = mod(p, period);
p = min(p, period - p);
idx = p + 1;
end
