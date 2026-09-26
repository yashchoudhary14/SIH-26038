function rgb = readImage(src)
%READIMAGE Load a fundus photograph as H x W x 3 uint8 RGB.
%
%   Accepts a file path or an array. Greyscale is replicated to three
%   channels, an alpha channel is dropped, indexed images are converted, and
%   16-bit images are scaled to 8 bits -- what cv2.imread(IMREAD_COLOR) does.

if ischar(src) || isstring(src)
    src = char(src);
    if ~isfile(src)
        error('drscreen:io:notFound', 'Cannot read image: %s', src);
    end
    [rgb, map] = imread(src);
    if ~isempty(map)
        rgb = im2uint8(ind2rgb(rgb, map));
    end
else
    rgb = src;
end
if size(rgb, 3) == 4
    rgb = rgb(:, :, 1:3);
elseif size(rgb, 3) == 1
    rgb = repmat(rgb, 1, 1, 3);
end
if ~isa(rgb, 'uint8')
    rgb = im2uint8(rgb);
end
end
