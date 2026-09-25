function P = paths()
%PATHS Every directory the code reads from or writes to, in one place.
%
%   Works from the source tree and from a MATLAB Compiler build. Inside a
%   compiled app the read-only resources (models, web, data) are unpacked
%   under ctfroot; if they are not at the source-tree position relative to
%   this file they are located by searching for the model spec. Outputs go to
%   a per-user folder there, since the package directory may be read-only.

persistent cache
if ~isempty(cache)
    P = cache;
    return
end

here = fileparts(mfilename('fullpath'));      % .../+drscreen
root = fileparts(here);
if isdeployed && ~isfile(fullfile(root, 'models', 'segmentation_spec.json'))
    hit = dir(fullfile(ctfroot, '**', 'segmentation_spec.json'));
    if ~isempty(hit)
        root = fileparts(hit(1).folder);
    end
end

P.root = root;
P.models = fullfile(root, 'models');
% The website, and the project's README.md and RESULTS.md (served at /README.md
% and /RESULTS.md). Run from inside the project repository, serve the
% repository's own copies -- the ones everyone edits -- so the MATLAB backend
% never serves a stale one. matlab/web and matlab/docs are the packaged copies a
% compiled app ships; deploy/sync_web.m writes them from the repository, and
% neither is committed.
P.web = fullfile(root, 'web');
P.webPackaged = P.web;
P.docs = fullfile(root, 'docs');
repo = fileparts(root);
if ~isdeployed && isfile(fullfile(repo, 'web', 'index.html')) ...
        && isfolder(fullfile(repo, 'src', 'drscreen'))
    P.web = fullfile(repo, 'web');
    P.docs = repo;
end
P.data = fullfile(root, 'data');
P.vset = fullfile(P.data, 'verification_set');
P.showcase = fullfile(P.data, 'showcase');
P.tests = fullfile(root, 'tests');

if isdeployed
    base = getenv('LOCALAPPDATA');
    if isempty(base), base = tempdir; end
    P.outputs = fullfile(base, 'DRScreen');
else
    P.outputs = fullfile(root, 'outputs');
end
P.reports = fullfile(P.outputs, 'reports');
P.audit = fullfile(P.outputs, 'audit', 'reviews.jsonl');
cache = P;
end
