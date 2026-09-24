function build_standalone(target)
%BUILD_STANDALONE Package the screening console and/or web server as executables.
%
%   build_standalone            % both
%   build_standalone('console') % desktop screening console  -> dist/console/
%   build_standalone('web')     % web console + REST server  -> dist/web/
%
%   Requires MATLAB Compiler. The machines that run the result need only the
%   free MATLAB Runtime of the same release (the installer built here can
%   download it), not a MATLAB licence -- which is what makes a "local running
%   setup" at a screening camp practical.
%
%   The exported weights, the web console, the sample photographs and the docs
%   are packaged alongside the code; outputs (reports, the audit log) are
%   written under %LOCALAPPDATA%\DRScreen on the target machine.

if nargin < 1 || isempty(target), target = 'all'; end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root, 'app'));
assert(~isempty(which('compiler.build.standaloneApplication')), ...
    'MATLAB Compiler is required to build standalone applications.');
sync_web();          % ship the repository's current website, not a stale copy

extra = {fullfile(root, 'models'), fullfile(root, 'web'), fullfile(root, 'data'), ...
         fullfile(root, 'docs'), fullfile(root, 'README.md'), ...
         fullfile(root, '+drscreen', '+report', 'report.css')};
dist = fullfile(root, 'dist');

if any(strcmp(target, {'all', 'console'}))
    opts = {'ExecutableName', 'DRScreenConsole', 'AdditionalFiles', extra, ...
            'OutputDir', fullfile(dist, 'console'), 'Verbose', 'on'};
    if ispc
        r = compiler.build.standaloneWindowsApplication(fullfile(root, 'launch_console.m'), opts{:});
    else
        r = compiler.build.standaloneApplication(fullfile(root, 'launch_console.m'), opts{:});
    end
    fprintf('console -> %s\n', r.Options.OutputDir);
    installer(r, 'DRScreenConsole', fullfile(dist, 'console_installer'));
end

if any(strcmp(target, {'all', 'web'}))
    r = compiler.build.standaloneApplication(fullfile(root, 'launch_web.m'), ...
        'ExecutableName', 'DRScreenWeb', 'AdditionalFiles', extra, ...
        'OutputDir', fullfile(dist, 'web'), 'Verbose', 'on');
    fprintf('web server -> %s\n', r.Options.OutputDir);
    installer(r, 'DRScreenWeb', fullfile(dist, 'web_installer'));
end
end


function installer(buildResults, name, outDir)
% An installer that fetches the MATLAB Runtime on the target machine if needed.
try
    compiler.package.installer(buildResults, 'ApplicationName', name, ...
        'InstallerName', [name 'Installer'], 'OutputDir', outDir, 'RuntimeDelivery', 'web');
    fprintf('installer -> %s\n', outDir);
catch err
    warning('drscreen:deploy', 'Installer not built (%s). The executable above still runs where the MATLAB Runtime is installed.', err.message);
end
end
