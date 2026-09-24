function [grade, reasons] = ruleGrade(F)
%RULEGRADE Apply the International Clinical DR Severity Scale directly.
%
%   The criteria strings are what the report prints as its justification, and
%   the rule grade is cross-checked against the network's grade in every
%   result. This is also the classical-CV arm of the ablation.

reasons = {};
ma = F.counts.microaneurysm;
he = F.counts.hemorrhage;
ex = F.counts.hard_exudate;
se = F.counts.soft_exudate;
nv = F.counts.neovascularization;

% --- grade 4: proliferative DR ---------------------------------------------
% If NV was never assessed, say so: this arm is the only route to grade 4, and
% a rule engine that cannot reach its own top grade must declare it, or a
% "grade 3" reads as "assessed and not proliferative".
if ismember('neovascularization', F.unassessed)
    reasons{end + 1} = ['Neovascularisation NOT ASSESSED - no pixel supervision ' ...
        'for this class in the training corpus, so proliferative DR cannot be ' ...
        'excluded by lesion evidence.'];
elseif nv > 0 || F.nv_at_disc || F.nv_elsewhere
    where = {};
    if F.nv_at_disc, where{end + 1} = 'at the disc (NVD)'; end
    if F.nv_elsewhere, where{end + 1} = 'elsewhere (NVE)'; end
    reasons{end + 1} = [strtrim(['Neovascularisation detected ' strjoin(where, ' and ')]), ...
        ' - defines proliferative DR.'];
    grade = 4;
    return
end

% --- grade 3: severe NPDR, the 4-2-1 rule ----------------------------------
severe = false;
if F.quadrants_with_hemorrhage >= 4
    reasons{end + 1} = sprintf(['20 or more haemorrhages/microaneurysms in each of ' ...
        '%d quadrants (4-2-1 rule, ''4'').'], F.quadrants_with_hemorrhage);
    severe = true;
end
if F.quadrants_with_beading >= 2
    reasons{end + 1} = sprintf('Definite venous beading in %d quadrants (4-2-1 rule, ''2'').', ...
        F.quadrants_with_beading);
    severe = true;
end
if severe
    grade = 3;
    return
end

% --- grade 2: moderate NPDR ------------------------------------------------
if he > 0 || ex > 0 || se > 0 || ma >= 8
    bits = {};
    if ma, bits{end + 1} = sprintf('%d microaneurysm%s', ma, plural(ma)); end
    if he, bits{end + 1} = sprintf('%d haemorrhage%s', he, plural(he)); end
    if ex, bits{end + 1} = sprintf('%d hard-exudate focus/foci', ex); end
    if se, bits{end + 1} = sprintf('%d cotton-wool spot%s', se, plural(se)); end
    reasons{end + 1} = ['More than microaneurysms alone but less than severe NPDR: ' ...
        strjoin(bits, ', ') '.'];
    grade = 2;
    return
end

% --- grade 1: mild NPDR ----------------------------------------------------
if ma > 0
    reasons{end + 1} = sprintf('%d microaneurysm%s only - mild NPDR.', ma, plural(ma));
    grade = 1;
    return
end

reasons{end + 1} = ['No microaneurysms, haemorrhages, exudates or ' ...
    'neovascularisation detected.'];
grade = 0;
end


function s = plural(n)
if n ~= 1
    s = 's';
else
    s = '';
end
end
