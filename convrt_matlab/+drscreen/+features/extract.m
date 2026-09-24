function F = extract(lesionProbs, lm, fovMask, vesselMask, thresholds, unassessed)
%EXTRACT Turn lesion probability maps into the quantities ICDR is defined in.
%
%   F = drscreen.features.extract(probs, landmarks, fovMask, vesselMask, thr, unassessed)
%
%   Lesion counts per retinal quadrant, neovascularisation location, and
%   distance from the fovea in disc diameters -- what lets the report say
%   "27 haemorrhages across 4 quadrants" instead of pointing at a heatmap.
%
%   thresholds : scalar, or a struct of per-class cut-points. A blanket 0.5 is
%                an unfitted parameter: fitted on held-out IDRiD the optimum is
%                0.6-0.95, and at 0.5 the exudate channel tripped the
%                macular-oedema rule on healthy retinas.
%   unassessed : classes the segmentation model was never trained on, so a
%                count of zero for them means "not assessed", not "absent".

C = drscreen.constants();
if nargin < 5 || isempty(thresholds), thresholds = 0.5; end
if nargin < 6, unassessed = {}; end

[h, w, ~] = size(lesionProbs);
scale = (h * w) / (512 * 512);
if ~isempty(fovMask)
    retinaArea = max(nnz(fovMask > 0), 1);
else
    retinaArea = h * w;
end

F.counts = struct();
F.area_fraction = struct();
F.per_quadrant = struct();
for i = 1:numel(C.LESION_CLASSES)
    F.per_quadrant.(C.LESION_CLASSES{i}) = cell2struct(num2cell(zeros(1, 4)), C.QUADRANTS, 2);
end
F.quadrants_with_hemorrhage = 0;
F.quadrants_with_beading = 0;
F.nv_at_disc = false;
F.nv_elsewhere = false;
F.unassessed = cellstr(unassessed);
F.lesions_within_1dd_of_fovea = 0;
F.exudates_within_1dd_of_fovea = 0;
F.nearest_lesion_dd = 99.0;
F.vessel_density = 0.0;
F.vessel_caliber_cv = 0.0;
F.instances = struct('lesion', {}, 'x', {}, 'y', {}, 'area_px', {}, ...
                     'quadrant', {}, 'dd_from_fovea', {}, 'confidence', {});

nearest = 99.0;
for ci = 1:numel(C.LESION_CLASSES)
    name = C.LESION_CLASSES{ci};
    if isstruct(thresholds)
        thr = 0.5;
        if isfield(thresholds, name), thr = thresholds.(name); end
    else
        thr = thresholds;
    end
    prob = lesionProbs(:, :, ci);
    binary = prob >= thr;
    if ~isempty(fovMask)
        binary = binary & (fovMask > 0);
    end
    F.area_fraction.(name) = nnz(binary) / retinaArea;

    minArea = max(1, fix(C.MIN_AREA_512.(name) * scale));
    S = drscreen.cv.conncomp(binary);
    count = 0;
    for k = 1:numel(S)
        a = S(k).Area;
        if a < minArea
            continue
        end
        cx = S(k).Centroid(1);
        cy = S(k).Centroid(2);
        q = drscreen.preprocess.quadrantOf(cx, cy, lm);
        dd = drscreen.preprocess.ddFromFovea(cx, cy, lm);
        conf = mean(double(prob(S(k).PixelIdxList)));
        F.instances(end + 1) = struct('lesion', name, 'x', cx, 'y', cy, 'area_px', a, ...
            'quadrant', q, 'dd_from_fovea', dd, 'confidence', conf);
        F.per_quadrant.(name).(q) = F.per_quadrant.(name).(q) + 1;
        count = count + 1;
        nearest = min(nearest, dd);
        if dd <= 1.0
            F.lesions_within_1dd_of_fovea = F.lesions_within_1dd_of_fovea + 1;
            if strcmp(name, 'hard_exudate')
                F.exudates_within_1dd_of_fovea = F.exudates_within_1dd_of_fovea + 1;
            end
        end
        if strcmp(name, 'neovascularization')
            if hypot(cx - lm.disc_xy(1), cy - lm.disc_xy(2)) <= 1.5 * lm.disc_radius
                F.nv_at_disc = true;
            else
                F.nv_elsewhere = true;
            end
        end
    end
    F.counts.(name) = count;
end
F.nearest_lesion_dd = nearest;

% 4-2-1 rule, the "4": quadrants carrying >= 20 haemorrhages + microaneurysms
heavy = 0;
for qi = 1:numel(C.QUADRANTS)
    q = C.QUADRANTS{qi};
    if F.per_quadrant.hemorrhage.(q) + F.per_quadrant.microaneurysm.(q) >= 20
        heavy = heavy + 1;
    end
end
F.quadrants_with_hemorrhage = heavy;

if ~isempty(vesselMask)
    F.vessel_density = nnz(vesselMask > 0) / retinaArea;
    [F.vessel_caliber_cv, F.quadrants_with_beading] = ...
        drscreen.features.caliberStats(vesselMask, lm);
end
end
