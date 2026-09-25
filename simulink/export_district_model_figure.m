function export_district_model_figure(outDir)
%EXPORT_DISTRICT_MODEL_FIGURE  Render district_model.slx for slides and print.
%
%   Runs the model once (so the displays show a session's counts), then
%   writes figures/district_model.png (300 dpi), .svg and .pdf (vector).
%   Also builds the recapture variant (include_gate = 1) as
%   figures/district_model_recapture.slx and renders it the same way; the
%   dossier draws whichever one matches its recapture switch.

here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outDir), outDir = fullfile(here, 'figures'); end
if ~isfolder(outDir), mkdir(outDir); end
[p, PatientBus] = district_model_params();
assignin('base', 'PatientBus', PatientBus);

render('district_model', fullfile(here, 'district_model.slx'), p);

q = p; q.include_gate = 1;
file = fullfile(outDir, 'district_model_recapture.slx');
build_district_model(q, file);
render('district_model_recapture', file, q);

    function render(mdl, file, p)
        assignin('base', 'p', p);
        if bdIsLoaded(mdl), close_system(mdl, 0); end
        load_system(file);
        sim(mdl);
        base = fullfile(outDir, mdl);
        print(['-s' mdl], '-dpng', '-r300', [base '.png']);
        print(['-s' mdl], '-dsvg', [base '.svg']);
        print(['-s' mdl], '-dpdf', [base '.pdf']);
        close_system(mdl, 0);
        fprintf('wrote %s.{png,svg,pdf}\n', base);
    end
end
