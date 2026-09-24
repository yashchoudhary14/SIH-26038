function synced = sync_web()
%SYNC_WEB Refresh convrt_matlab/web from the project repository's web/.
%
%   sync_web
%
%   The website is developed in the repository's web/ folder, and the MATLAB
%   server serves that folder directly when it runs inside the repository
%   (+drscreen/paths.m). convrt_matlab/web is the packaged copy that compiled
%   apps ship and that a stand-alone checkout of convrt_matlab uses; this makes
%   it an exact mirror of web/ (files removed there are removed here too).
%   build_standalone calls it before every build. Does nothing, and returns
%   false, when convrt_matlab is not inside the repository.

root = fileparts(fileparts(mfilename('fullpath')));
src = fullfile(fileparts(root), 'web');
dst = fullfile(root, 'web');
synced = isfile(fullfile(src, 'index.html'));
if ~synced
    fprintf('sync_web: no repository web/ next to %s; packaged copy left as is\n', root);
    return
end
if isfolder(dst)
    rmdir(dst, 's');
end
copyfile(src, dst);
fprintf('sync_web: %s -> %s\n', src, dst);
if nargout == 0, clear synced; end
end
