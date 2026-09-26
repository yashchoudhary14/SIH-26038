function s = readText(path)
%READTEXT Read a UTF-8 text file into a char row.
%
%   fileread uses the platform default encoding (windows-1252 on Windows), which
%   mangles the em dashes and multiplication signs in the case provenance
%   strings and the report text.
fid = fopen(path, 'r', 'n', 'UTF-8');
if fid < 0
    error('drscreen:io:read', 'Cannot open %s.', path);
end
closer = onCleanup(@() fclose(fid));
s = fread(fid, [1 Inf], '*char');
if ~isempty(s) && s(1) == char(65279)      % strip a UTF-8 BOM
    s = s(2:end);
end
end
