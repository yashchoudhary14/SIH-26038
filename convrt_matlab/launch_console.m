function app = launch_console(bundle)
%LAUNCH_CONSOLE Open the desktop screening console.
%
%   launch_console            % the default bundle, 'pooled' (DEFAULT_BUNDLE in +drscreen/constants.m)
%   launch_console('prepool') % grader trained on APTOS + IDRiD, Messidor-2 held out
%
%   Also the entry point of the standalone desktop app (deploy/build_standalone.m):
%   DRScreenConsole.exe [bundle]. The compiled app keeps a log in
%   %LOCALAPPDATA%\DRScreen\logs\console.log and shows startup failures in a
%   dialog.

root = fileparts(mfilename('fullpath'));
if ~isdeployed
    addpath(root, fullfile(root, 'app'));
end
if nargin < 1 || isempty(bundle), bundle = drscreen.constants().DEFAULT_BUNDLE; end
if ~isdeployed
    app = ScreeningConsole(bundle);
    return
end

logDir = fullfile(drscreen.paths().outputs, 'logs');
if ~isfolder(logDir), mkdir(logDir); end
diary(fullfile(logDir, 'console.log'));
cleaner = onCleanup(@() diary('off'));
fprintf('[%s] Starting the console with the %s bundle\n', char(datetime('now')), bundle);
try
    app = ScreeningConsole(bundle);
    fprintf('[%s] Ready\n', char(datetime('now')));
catch err
    fprintf('FAILED: %s\n', getReport(err, 'extended', 'hyperlinks', 'off'));
    f = uifigure('Name', 'DR Screening Console', 'Position', [100 100 480 180]);
    uialert(f, sprintf('%s\n\nLog: %s', err.message, fullfile(logDir, 'console.log')), ...
        'The console could not start', 'CloseFcn', @(~, ~) delete(f));
    waitfor(f);
    app = [];
    return
end
% A compiled app exits when its main function returns; keep it alive for as
% long as the console window is open.
app.waitUntilClosed();
end
