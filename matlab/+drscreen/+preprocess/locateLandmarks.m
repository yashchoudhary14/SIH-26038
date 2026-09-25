function lm = locateLandmarks(img, mask, vesselWeight)
%LOCATELANDMARKS Analytic optic-disc and fovea localisation.
%
%   Two ICDR criteria are geometric: severe NPDR is defined by lesion counts
%   per quadrant, CSME by distance from the fovea in disc diameters. Both need
%   a coordinate frame, and one that fails silently is worse than none, so this
%   is a closed-form detector that returns a confidence rather than a network.
%
%   Disc: brightest region at disc scale, weighted by vessel convergence -- the
%   disc is where every arcade meets, which separates it from a confluent
%   exudate (bright but avascular). Fovea: darkest point on a ring 1.6-3.4 disc
%   diameters from the disc, along the disc-to-retina-centre axis.
%
%   Coordinates are 0-based pixels in the standardised frame.

if nargin < 3 || isempty(vesselWeight)
    vesselWeight = drscreen.constants().DISC_VESSEL_WEIGHT;
end
[h, w, ~] = size(img);
if nargin < 2 || isempty(mask)
    mask = 255 * ones(h, w, 'uint8');
end
m = uint8(mask > 0);

if size(img, 3) == 3
    gray = drscreen.cv.rgb2gray(img);
else
    gray = uint8(img);
end
flat = drscreen.preprocess.flatField(gray, m);

% exclude a rim so the aperture edge cannot win either extremum
er = max(5, bitor(fix(0.06 * min(h, w)), 1));
inner = drscreen.cv.morph(m, 'erode', er);

% ---------------- optic disc ------------------------------------------
discR0 = 0.065 * min(h, w);
bright = drscreen.cv.gaussianBlur(flat, discR0 * 0.7);
vd = drscreen.preprocess.vesselDensity(img, m);
vdN = vd / max(max(vd(:)), 1e-6);
fin = flat(inner > 0);
q = drscreen.cv.percentile(fin, [5 99.5]);
lo = q(1); hi = q(2);
brightN = min(max((bright - lo) / max(hi - lo, 1e-6), 0), 1.5);

score = brightN .* ((1 - vesselWeight) + vesselWeight * vdN);
score(inner == 0) = -1;
[dy, dx, discPeak] = drscreen.cv.argmaxRowMajor(score);

% refine the radius: grow a bright region around the peak
thr = drscreen.cv.percentile(fin, 97);
bw = (flat > thr) & (inner > 0);
bw = drscreen.cv.morph(bw, 'close', bitor(fix(discR0 * 0.5), 1));
[S, L] = drscreen.cv.conncomp(bw);
discR = discR0;
if ~isempty(S) && L(dy + 1, dx + 1) > 0
    s = S(L(dy + 1, dx + 1));
    discR = min(max(sqrt(s.Area / pi), 0.5 * discR0), 1.8 * discR0);
    dx = fix(s.Centroid(1));
    dy = fix(s.Centroid(2));
end

bgLevel = drscreen.cv.percentile(score(inner > 0), 95);
discConf = min(max((discPeak - bgLevel) / max(bgLevel, 1e-3) * 3.0, 0), 1);

if dx < w / 2
    laterality = 'OS';
else
    laterality = 'OD';
end

% ---------------- fovea -----------------------------------------------
% Temporal direction = disc -> retina centroid. A left/right half-plane test
% flips sign on noise when the disc sits near the midline and sends the search
% nasally onto an arcade.
[yi, xi] = find(inner);
cxIn = mean(xi - 1);
cyIn = mean(yi - 1);
vx = cxIn - dx; vy = cyIn - dy;
nrm = hypot(vx, vy);
if nrm < 1e-3
    if dx > w / 2, vx = -1; else, vx = 1; end
    vy = 0; nrm = 1;
end
axisAng = atan2(vy / nrm, vx / nrm);

dd = 2 * discR;
smooth = drscreen.cv.gaussianBlur(flat, discR * 0.55);

% Darkness plus a weak prior on the textbook geometry (2.5 DD, on-axis). The
% prior only breaks ties; a genuinely displaced fovea still wins on darkness.
best = [];
bestVal = 1e9;
bestDark = 0;
for rMult = linspace(1.6, 3.4, 19)
    for ang = linspace(-0.62, 0.62, 25)
        fx = dx + dd * rMult * cos(axisAng + ang);
        fy = dy + dd * rMult * sin(axisAng + ang);
        xr = round(fx); yr = round(fy);
        if xr < 0 || xr >= w || yr < 0 || yr >= h || inner(yr + 1, xr + 1) == 0
            continue
        end
        penalty = 0.010 * abs(rMult - 2.5) + 0.020 * abs(ang);
        v = double(smooth(yr + 1, xr + 1)) + penalty;
        if v < bestVal
            bestVal = v;
            best = [xr, yr];
            bestDark = double(smooth(yr + 1, xr + 1));
        end
    end
end

if isempty(best)
    % disc near the rim: darkest point outside a 2-radius exclusion zone
    s2 = double(smooth);
    s2(inner == 0) = 1e9;
    [XX, YY] = meshgrid(0:w-1, 0:h-1);
    s2((XX - dx) .^ 2 + (YY - dy) .^ 2 <= fix(discR * 2) ^ 2) = 1e9;
    [fy2, fx2] = drscreen.cv.argmaxRowMajor(-s2);
    best = [fx2, fy2];
    bestDark = double(smooth(fy2 + 1, fx2 + 1));
end

surround = median(double(flat(inner > 0)));
foveaConf = min(max((surround - bestDark) / 0.18, 0), 1);

lm = struct( ...
    'disc_xy', [fix(dx), fix(dy)], ...
    'disc_radius', double(discR), ...
    'disc_confidence', round(double(discConf) * 1e4) / 1e4, ...
    'fovea_xy', [fix(best(1)), fix(best(2))], ...
    'fovea_confidence', round(double(foveaConf) * 1e4) / 1e4, ...
    'laterality', laterality, ...
    'disc_diameter_px', double(2 * discR));
end
