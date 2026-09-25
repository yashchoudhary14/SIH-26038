function paths = list()
%LIST Full paths of the committed sample photographs, in name order.
d = fullfile(drscreen.paths().vset, 'images');
paths = {};
if ~isfolder(d), return; end
files = [dir(fullfile(d, '*.jpg')); dir(fullfile(d, '*.jpeg')); dir(fullfile(d, '*.png'))];
names = sort({files.name});
paths = cellfun(@(n) fullfile(d, n), names, 'UniformOutput', false);
end
