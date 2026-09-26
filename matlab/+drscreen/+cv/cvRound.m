function r = cvRound(x)
%CVROUND Round to nearest, ties to even -- OpenCV's cvRound / saturate_cast.
%
%   MATLAB's round() sends ties away from zero; OpenCV's rounding (lrint under
%   the default FP mode) sends them to the even neighbour. The difference only
%   shows on exact .5 values, but those are not rare where a float32 product is
%   exact -- CLAHE's LUT is one such place -- so every OpenCV-compatible
%   primitive rounds through here.

r = round(x);
tie = abs(x - fix(x)) == 0.5;
if any(tie(:))
    r(tie) = 2 * round(x(tie) / 2);
end
end
