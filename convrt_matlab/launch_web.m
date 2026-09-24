function server = launch_web(varargin)
%LAUNCH_WEB Serve the web console and the REST API from MATLAB.
%
%   launch_web                                  % http://127.0.0.1:8000, foreground
%   s = launch_web('Background', true);         % keep using MATLAB; s.stop() to end
%   launch_web('Port', 8080, 'Bundle', 'pooled', 'Host', '0.0.0.0')   % reachable on the LAN
%
%   Bundle defaults to DEFAULT_BUNDLE in +drscreen/constants.m.
%
%   The page is the same web/index.html the Python FastAPI backend serves, and
%   the endpoints return the same JSON, so the site cannot tell which backend
%   it is talking to (except that /health reports runtime "matlab").
%
%   Also the entry point of the standalone web-server app: compiled, it opens a
%   small control window and serves until that window is closed. The compiled
%   app keeps a log in %LOCALAPPDATA%\DRScreen\logs\web_server.log, and shows
%   any startup failure in a dialog as well.

ip = inputParser;
ip.addParameter('Port', 8000);
ip.addParameter('Host', '127.0.0.1');
ip.addParameter('Bundle', '');
ip.addParameter('Background', false);
ip.addParameter('OpenBrowser', true);
ip.parse(varargin{:});
o = ip.Results;
% A compiled app receives its command-line options as text:
%   DRScreenWeb.exe Port 8080 Bundle pooled OpenBrowser false
o.Port = asNumber(o.Port);
o.Background = asLogical(o.Background);
o.OpenBrowser = asLogical(o.OpenBrowser);

root = fileparts(mfilename('fullpath'));
if ~isdeployed
    addpath(root);
    server = serve(o);
    return
end

% Compiled: there is no console to read, so log to a file and report failures
% in a dialog rather than exiting silently.
logDir = fullfile(drscreen.paths().outputs, 'logs');
if ~isfolder(logDir), mkdir(logDir); end
diary(fullfile(logDir, 'web_server.log'));
cleaner = onCleanup(@() diary('off'));
try
    server = serve(o);
catch err
    note('FAILED: %s', getReport(err, 'extended', 'hyperlinks', 'off'));
    f = uifigure('Name', 'DR Screening server', 'Position', [100 100 480 180]);
    uialert(f, sprintf('%s\n\nLog: %s', err.message, fullfile(logDir, 'web_server.log')), ...
        'The screening server could not start', 'CloseFcn', @(~, ~) delete(f));
    waitfor(f);
    server = [];
end
end


function server = serve(o)
if isempty(o.Bundle), o.Bundle = drscreen.constants().DEFAULT_BUNDLE; end
note('Loading the %s model bundle...', o.Bundle);
t = tic;
server = drscreen.server.Server('Port', o.Port, 'Host', o.Host, 'Bundle', o.Bundle);
note('Models loaded in %.1f s (convolutions: %s).', toc(t), ...
    ternary(drscreen.nn.useToolbox(), 'Deep Learning Toolbox', 'plain MATLAB'));
browseUrl = sprintf('http://%s:%d/', strrep(o.Host, '0.0.0.0', '127.0.0.1'), o.Port);

% Bind before anything else, so a busy port is reported plainly -- and the
% browser is never pointed at whatever else is listening there.
try
    server.open();
catch err
    if ~any(contains(err.message, {'in use', 'being used by another process', 'BindException'}))
        rethrow(err);
    end
    msg = sprintf(['Cannot serve on port %d: it is probably in use by another ' ...
        'program (another screening server?).\n\nStart on a free port instead, e.g.\n' ...
        '    launch_web(''Port'', 8080)\n    DRScreenWeb.exe Port 8080\n\n(%s)'], ...
        o.Port, err.message);
    error('drscreen:portInUse', '%s', msg);
end
note('Listening on %s', browseUrl);
note('Website: %s', drscreen.paths().web);

if o.Background && ~isdeployed
    server.start();
    if o.OpenBrowser, web(browseUrl, '-browser'); end
    return
end

% Foreground: a small control window, so closing it (or Ctrl+C) stops the server.
fig = uifigure('Name', 'DR Screening server', 'Position', [100 100 420 150]);
gl = uigridlayout(fig, [3 2], 'RowHeight', {'1x', 30, 30});
lbl = uilabel(gl, 'Text', sprintf('Serving %s\nmodel bundle: %s', browseUrl, o.Bundle), ...
    'WordWrap', 'on');
lbl.Layout.Column = [1 2];
uibutton(gl, 'Text', 'Open in browser', 'ButtonPushedFcn', @(~, ~) web(browseUrl, '-browser'));
uibutton(gl, 'Text', 'Stop server', 'ButtonPushedFcn', @(~, ~) delete(fig));
status = uilabel(gl, 'Text', 'Waiting for requests...', 'FontColor', [0.4 0.4 0.4]);
status.Layout.Column = [1 2];
note('Control window open; serving until it is closed.');
if o.OpenBrowser, web(browseUrl, '-browser'); end
server.run(@() ~isvalid(fig));
if isvalid(fig), delete(fig); end
note('Stopped.');
end


function note(fmt, varargin)
fprintf(['[%s] ' fmt '\n'], char(datetime('now', 'Format', 'HH:mm:ss')), varargin{:});
end


function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end


function v = asNumber(v)
if ischar(v) || isstring(v), v = str2double(v); end
end


function v = asLogical(v)
if ischar(v) || isstring(v), v = any(strcmpi(v, {'true', '1', 'yes', 'on'})); end
v = logical(v);
end
