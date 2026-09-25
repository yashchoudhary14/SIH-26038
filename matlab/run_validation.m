function T = run_validation(bundle)
%RUN_VALIDATION Screen all 72 committed real photographs; report metrics and
%agreement with the Python pipeline.
%
%   T = run_validation            % the default bundle (constants.m)
%   T = run_validation('pooled')
%
%   Two things are measured, and they should not be confused:
%
%   1. Port fidelity -- does MATLAB return the same grade, decision and
%      P(referable) as the Python pipeline on the same photograph? Both run
%      with MC dropout off (deterministic) against tests/fixtures/cases.json.
%      This is the number that says the conversion is correct.
%
%   2. Screening metrics on these 72 images. The 60 showcase images were
%      SELECTED because the model handles them well, and the 12 verification
%      cases are a curated demo set, so these numbers are demonstrations, not
%      validation. The measured sensitivity and specificity -- on 631 internal
%      and 1,744 Messidor-2 images -- are in RESULTS.md at the repository root.

root = fileparts(mfilename('fullpath'));
addpath(root);
if nargin < 1 || isempty(bundle), bundle = drscreen.constants().DEFAULT_BUNDLE; end
P = drscreen.paths();
fx = fullfile(P.tests, 'fixtures', 'cases.json');
ref = jsondecode(drscreen.io.readText(fx));
rows = ref.cases;
pipe = drscreen.Pipeline.load(bundle, 'McSamples', 0, 'EnableCam', false);

n = numel(rows);
name = strings(n, 1); set = strings(n, 1);
truth = nan(n, 1); grade = nan(n, 1); pyGrade = nan(n, 1);
dec = strings(n, 1); pyDec = strings(n, 1); urg = strings(n, 1);
pref = nan(n, 1); pyPref = nan(n, 1);
fprintf('Screening %d photographs with bundle %s (MC dropout off)\n', n, bundle);
for i = 1:n
    if iscell(rows), r = rows{i}; else, r = rows(i); end
    py = r.(bundle);
    res = pipe.run(fullfile(P.root, strrep(r.file, '/', filesep)), r.case, false);
    name(i) = r.case; set(i) = r.set; truth(i) = r.true_grade;
    grade(i) = res.grade; dec(i) = res.decision; urg(i) = res.urgency;
    pref(i) = res.referable_probability;
    pyGrade(i) = py.grade; pyDec(i) = py.decision; pyPref(i) = py.referable_probability;
    mark = '';
    if grade(i) ~= pyGrade(i) || dec(i) ~= pyDec(i), mark = '   <-- differs from Python'; end
    fprintf('  %-16s true %d  MATLAB %d %-15s P %.4f | Python %d %-15s P %.4f%s\n', ...
        r.case, truth(i), grade(i), dec(i), pref(i), pyGrade(i), pyDec(i), pyPref(i), mark);
end

T = table(name, set, truth, grade, dec, urg, pref, pyGrade, pyDec, pyPref, ...
    'VariableNames', {'case', 'set', 'true_grade', 'grade', 'decision', 'urgency', ...
    'p_referable', 'python_grade', 'python_decision', 'python_p_referable'});

fprintf('\n--- Port fidelity (MATLAB vs Python, same images, deterministic) ---\n');
fprintf('  grade identical        %d / %d\n', sum(grade == pyGrade), n);
fprintf('  decision identical     %d / %d\n', sum(dec == pyDec), n);
d = abs(pref - pyPref);
fprintf('  |P(referable) diff|    median %.2e, max %.2e\n', median(d), max(d));

fprintf('\n--- Screening metrics on these 72 curated images (demonstration only) ---\n');
g = ~isnan(grade) & grade >= 0;
refTrue = truth >= 2;
refPred = dec ~= "auto_report" & dec ~= "recapture";
fprintf('  gradeable              %d / %d\n', sum(g), n);
fprintf('  exact grade            %d / %d\n', sum(grade(g) == truth(g)), sum(g));
fprintf('  within one grade       %d / %d\n', sum(abs(grade(g) - truth(g)) <= 1), sum(g));
fprintf('  referable referred     %d / %d\n', sum(refPred & refTrue), sum(refTrue));
fprintf('  non-referable not referred %d / %d\n', sum(~refPred & ~refTrue), sum(~refTrue));
st = truth >= 3;
fprintf('  sight-threatening urgent %d / %d\n', sum(urg(st) == "urgent"), sum(st));
fprintf('\nMeasured sensitivity/specificity (631 internal, 1,744 Messidor-2): RESULTS.md at the repository root\n');

out = fullfile(P.outputs, sprintf('validation_%s.csv', bundle));
if ~isfolder(P.outputs), mkdir(P.outputs); end
writetable(T, out);
fprintf('Table written to %s\n', out);
if nargout == 0, clear T; end
end
