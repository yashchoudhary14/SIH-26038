function ok = check_install()
%CHECK_INSTALL Report what this MATLAB installation can run.
%
%   Required:  MATLAB R2021a or newer (jsonencode PrettyPrint), .NET on
%              Windows or Java elsewhere (the web server socket), and Image Processing
%              Toolbox (morphology, connected components, imread helpers).
%   Optional:  Deep Learning Toolbox (used automatically to run the
%              convolutions ~3x faster end to end; results are the same),
%              MATLAB Compiler (standalone desktop/web apps),
%              Simulink + SimEvents (the telemedicine capacity model).
%   Not needed: the ONNX converter -- both networks run from exported weights.

rel = version('-release');
fprintf('MATLAB R%s\n', rel);
ok = true;
okRel = str2double(rel(1:4)) >= 2021;            % R2021a or newer
ok = report('MATLAB R2021a or newer', okRel, true) && ok;
% The web server's socket: .NET on Windows (the only option in a compiled
% app, whose runtime has no JVM), Java elsewhere.
if ispc
    ok = report('.NET (web server transport)', NET.isNETSupported, true) && ok;
else
    ok = report('Java (web server transport)', usejava('jvm'), true) && ok;
end
ipt = license('test', 'Image_Toolbox') && exist('imerode', 'file') > 0;
ok = report('Image Processing Toolbox', ipt, true) && ok;
report('Deep Learning Toolbox (faster convolutions)', drscreen.nn.useToolbox(), false);
report('MATLAB Compiler (standalone apps)', license('test', 'Compiler') && ...
    ~isempty(which('compiler.build.standaloneApplication')), false);
report('Simulink', license('test', 'Simulink') && exist('simulink', 'file') > 0, false);
report('SimEvents', license('test', 'SimEvents'), false);

P = drscreen.paths();
ok = report('exported weights (models/)', isfile(fullfile(P.models, 'segmentation.mat')) && ...
    isfile(fullfile(P.models, 'prepool', 'grader.mat')), true) && ok;
report('parity fixtures (tests/fixtures/)', isfile(fullfile(P.tests, 'fixtures', 'cases.json')), false);
report('sample photographs (data/verification_set/)', ~isempty(drscreen.samples.list()), false);
if ok
    fprintf('\nReady. Try: launch_console  or  launch_web  or  run_tests\n');
else
    fprintf('\nSomething required is missing (see above).\n');
end
if nargout == 0, clear ok; end
end


function ok = report(name, ok, required)
if ok
    mark = 'ok  ';
elseif required
    mark = 'MISS';
else
    mark = '--  ';
end
tag = '';
if ~required, tag = ' (optional)'; end
fprintf('  [%s] %s%s\n', mark, name, tag);
end
