function synced = sync_web()
%SYNC_WEB Refresh the packaged website and project documents from the repository.
%
%   sync_web
%
%   The website is developed in the repository's web/ folder, and the MATLAB
%   server serves that folder -- and the repository's README.md and RESULTS.md
%   -- directly when it runs inside the repository (+drscreen/paths.m). A
%   compiled app cannot reach the repository, so it ships copies: matlab/web,
%   an exact mirror of web/ (files removed there are removed here too), and
%   matlab/docs, holding README.md, RESULTS.md and DATASETS.md. Neither copy
%   is committed. build_standalone calls this before every build. Does
%   nothing, and returns false, when this folder is not inside the repository.

root = fileparts(fileparts(mfilename('fullpath')));
repo = fileparts(root);
src = fullfile(repo, 'web');
synced = isfile(fullfile(src, 'index.html'));
if ~synced
    fprintf('sync_web: no repository web/ next to %s; packaged copies left as they are\n', root);
    if nargout == 0, clear synced; end
    return
end

dst = fullfile(root, 'web');
if isfolder(dst)
    rmdir(dst, 's');
end
copyfile(src, dst);
fprintf('sync_web: %s -> %s\n', src, dst);

docs = fullfile(root, 'docs');
if ~isfolder(docs)
    mkdir(docs);
end
for f = {fullfile(repo, 'README.md'), fullfile(repo, 'RESULTS.md'), fullfile(repo, 'docs', 'DATASETS.md')}
    copyfile(f{1}, docs);
end
fprintf('sync_web: README.md, RESULTS.md, DATASETS.md -> %s\n', docs);
if nargout == 0, clear synced; end
end
