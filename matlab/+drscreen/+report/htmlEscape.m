function s = htmlEscape(s)
%HTMLESCAPE Escape &, <, >, " and ' for safe inclusion in HTML.
s = char(string(s));
s = strrep(s, '&', '&amp;');
s = strrep(s, '<', '&lt;');
s = strrep(s, '>', '&gt;');
s = strrep(s, '"', '&quot;');
s = strrep(s, '''', '&#x27;');
end
