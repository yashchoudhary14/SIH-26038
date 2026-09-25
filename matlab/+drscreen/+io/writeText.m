function writeText(path, text, mode)
%WRITETEXT Write (or append) UTF-8 text, creating the folder if needed.
if nargin < 3, mode = 'w'; end
d = fileparts(path);
if ~isempty(d) && ~isfolder(d)
    mkdir(d);
end
fid = fopen(path, mode, 'n', 'UTF-8');
if fid < 0
    error('drscreen:io:write', 'Cannot open %s for writing.', path);
end
closer = onCleanup(@() fclose(fid));
fwrite(fid, unicode2native(char(text), 'UTF-8'));
end
