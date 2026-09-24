function s = maculaScore(lm, imageSize)
%MACULASCORE Quality criterion: is the macula in frame with usable margin?
%
%   At least one disc diameter of retina around the fovea is wanted, blended
%   with how clearly the fovea was found.

h = imageSize(1); w = imageSize(2);
fx = lm.fovea_xy(1); fy = lm.fovea_xy(2);
marginPx = min([fx, fy, w - fx, h - fy]);
margin = min(max(marginPx / max(lm.disc_diameter_px, 1e-6), 0), 1.5) / 1.5;
s = min(max(0.55 * margin + 0.45 * lm.fovea_confidence, 0), 1);
end
