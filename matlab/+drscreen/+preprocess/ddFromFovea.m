function d = ddFromFovea(x, y, lm)
%DDFROMFOVEA Distance from the fovea in disc diameters -- the CSME yardstick.
d = hypot(x - lm.fovea_xy(1), y - lm.fovea_xy(2)) / max(lm.disc_diameter_px, 1e-6);
end
