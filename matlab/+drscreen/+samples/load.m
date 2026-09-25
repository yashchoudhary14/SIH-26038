function [img, name, grade] = load(nameOrPath, grade)
%LOAD One of the committed real held-out photographs, as RGB uint8.
%
%   [img, name, grade] = drscreen.samples.load()                  % first case
%   [img, name, grade] = drscreen.samples.load('', 3)             % a grade-3 case
%   [img, name, grade] = drscreen.samples.load('grade3_case1')
%
%   Twelve APTOS-2019 and IDRiD photographs from the held-out test splits --
%   never trained on, never used to fit a threshold -- covering all five ICDR
%   grades. They stand in wherever "a fundus image" is needed without a
%   dataset download: the demo, the benchmark, the tests.

if nargin < 1, nameOrPath = ''; end
if nargin < 2, grade = []; end
paths = drscreen.samples.list();
if isempty(paths)
    error('drscreen:samples:missing', ['No sample photographs under %s. They ' ...
        'ship with the repository (data/verification_set/images).'], ...
        fullfile(drscreen.paths().vset, 'images'));
end
names = cellfun(@stemOf, paths, 'UniformOutput', false);

if ~isempty(nameOrPath) && isfile(nameOrPath)
    f = nameOrPath;
elseif ~isempty(nameOrPath)
    k = find(strcmp(names, nameOrPath), 1);
    if isempty(k)
        error('drscreen:samples:missing', 'No sample named "%s".', nameOrPath);
    end
    f = paths{k};
elseif ~isempty(grade) && ~isnan(grade)
    k = find(startsWith(names, sprintf('grade%d_', grade)), 1);
    if isempty(k)
        error('drscreen:samples:missing', 'No sample photograph of grade %d.', grade);
    end
    f = paths{k};
else
    f = paths{1};
end
img = drscreen.io.readImage(f);
name = stemOf(f);
tok = regexp(name, '^grade(\d)_', 'tokens', 'once');
if isempty(tok)
    grade = [];
else
    grade = str2double(tok{1});
end
end


function s = stemOf(f)
[~, s] = fileparts(f);
end
