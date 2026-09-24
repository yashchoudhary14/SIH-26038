function tf = useToolbox(setting)
%USETOOLBOX Whether convolutions run on the Deep Learning Toolbox, if present.
%
%   tf = drscreen.nn.useToolbox()        % current setting
%   drscreen.nn.useToolbox(false)        % force the plain-MATLAB path
%   drscreen.nn.useToolbox(true)         % use the toolbox when it is installed
%
%   The Deep Learning Toolbox is NOT required. When it is installed, dlconv
%   (oneDNN underneath) runs the convolutions 3-7x faster than the plain
%   shifted-GEMM code and agrees with it to ~1e-7 relative, so it is used
%   automatically. Availability is decided by calling dlconv once rather than
%   by which()/license(), because both misreport inside compiled apps.
%   Setting the environment variable DRSCREEN_PLAIN=1 forces the plain path
%   (the tests use this to exercise both).

persistent state
if nargin > 0
    state = logical(setting) && available();
elseif isempty(state)
    state = ~strcmp(getenv('DRSCREEN_PLAIN'), '1') && available();
end
tf = state;
end


function ok = available()
try
    y = dlconv(dlarray(ones(3, 3, 1, 'single'), 'SSC'), ones(1, 1, 1, 1, 'single'), 0);
    ok = size(y, 1) == 3 && size(y, 2) == 3;
catch
    ok = false;
end
end
