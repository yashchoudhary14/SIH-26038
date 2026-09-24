function s = base64Image(img, format, quality)
%BASE64IMAGE Encode an image as base64 PNG or JPEG (no data: prefix).
if nargin < 2, format = 'png'; end
if nargin < 3, quality = 88; end
tmp = [tempname '.' format];
cleaner = onCleanup(@() deleteIfExists(tmp));
if strcmpi(format, 'jpg') || strcmpi(format, 'jpeg')
    imwrite(img, tmp, 'jpg', 'Quality', quality);
else
    imwrite(img, tmp, format);
end
fid = fopen(tmp, 'r');
bytes = fread(fid, Inf, '*uint8');
fclose(fid);
s = matlab.net.base64encode(bytes');
end


function deleteIfExists(f)
if isfile(f)
    delete(f);
end
end
