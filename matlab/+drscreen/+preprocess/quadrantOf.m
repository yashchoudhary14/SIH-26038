function q = quadrantOf(x, y, lm)
%QUADRANTOF Retinal quadrant of a point, for the 4-2-1 severe-NPDR rule.
%
%   Quadrants are defined about the fovea, with the horizontal raphe dividing
%   superior from inferior and the fovea-disc axis dividing nasal from
%   temporal (ETDRS convention). Coordinates are 0-based pixels.

fx = lm.fovea_xy(1); fy = lm.fovea_xy(2);
ax = atan2(lm.disc_xy(2) - fy, lm.disc_xy(1) - fx);    % fovea -> disc == nasal
a = atan2(y - fy, x - fx) - ax;
a = mod(a + pi, 2 * pi) - pi;
if a >= -pi / 4 && a < pi / 4
    q = 'nasal';
elseif a >= pi / 4 && a < 3 * pi / 4
    q = 'inferior';
elseif a >= -3 * pi / 4 && a < -pi / 4
    q = 'superior';
else
    q = 'temporal';
end
end
