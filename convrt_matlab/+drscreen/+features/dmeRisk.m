function [risk, reason] = dmeRisk(F)
%DMERISK Macular-oedema risk from exudate proximity to the fovea (IDRiD scale).
%   0 no apparent oedema, 1 exudates present but > 1 DD from the fovea,
%   2 exudates within 1 DD (clinically significant; urgent referral).
if F.exudates_within_1dd_of_fovea > 0
    risk = 2;
    reason = sprintf(['%d hard-exudate focus/foci within 1 disc diameter of the ' ...
        'fovea - clinically significant macular oedema likely; urgent referral.'], ...
        F.exudates_within_1dd_of_fovea);
elseif F.counts.hard_exudate > 0
    risk = 1;
    reason = sprintf('Hard exudates present but the nearest is %.1f DD from the fovea.', ...
        F.nearest_lesion_dd);
else
    risk = 0;
    reason = 'No hard exudates detected in the macular region.';
end
end
