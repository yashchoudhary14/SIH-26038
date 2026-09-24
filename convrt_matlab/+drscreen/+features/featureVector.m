function v = featureVector(F)
%FEATUREVECTOR The 39-number clinical vector the fusion arm of the grader reads.
%
%   5 classes x (log1p count, area %) + 5 classes x 4 quadrants log1p counts
%   + 9 clinical scalars. The deployed image-only arm does not consume it; it is
%   reported for audit and kept identical to ClinicalFeatures.to_vector().
%   Counts are log1p-compressed: 0 vs 5 microaneurysms is clinically decisive,
%   60 vs 65 is not.

C = drscreen.constants();
v = [];
for i = 1:numel(C.LESION_CLASSES)
    c = C.LESION_CLASSES{i};
    v(end + 1) = log1p(F.counts.(c)); %#ok<AGROW>
    v(end + 1) = F.area_fraction.(c) * 100; %#ok<AGROW>
end
for i = 1:numel(C.LESION_CLASSES)
    c = C.LESION_CLASSES{i};
    for j = 1:numel(C.QUADRANTS)
        v(end + 1) = log1p(F.per_quadrant.(c).(C.QUADRANTS{j})); %#ok<AGROW>
    end
end
v = [v, F.quadrants_with_hemorrhage, F.quadrants_with_beading, ...
     double(F.nv_at_disc), double(F.nv_elsewhere), ...
     log1p(F.lesions_within_1dd_of_fovea), log1p(F.exudates_within_1dd_of_fovea), ...
     min(max(F.nearest_lesion_dd, 0), 10) / 10, F.vessel_density, F.vessel_caliber_cv];
v = single(v(:));
end
