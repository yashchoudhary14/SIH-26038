function out = saturateU8(x)
%SATURATEU8 OpenCV saturate_cast<uchar>: round ties-to-even, clamp to [0,255].
out = uint8(min(max(drscreen.cv.cvRound(double(x)), 0), 255));
end
