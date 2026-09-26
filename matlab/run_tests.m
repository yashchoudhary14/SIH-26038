function results = run_tests(scope)
%RUN_TESTS Run the parity test suite against the Python pipeline's fixtures.
%
%   run_tests                 % everything
%   run_tests('fast')         % primitives + networks only (about a minute)
%
%   The suite is layered narrow-to-wide so a disagreement is located where it
%   starts: OpenCV-compatible primitives (TestCv), the two networks on fixed
%   probes (TestModels), each preprocessing stage on real photographs
%   (TestPreprocess), the full pipeline on all 72 committed photographs
%   (TestPipelineParity), and the HTTP API (TestServer).

if nargin < 1, scope = 'all'; end
root = fileparts(mfilename('fullpath'));
addpath(root, fullfile(root, 'tests'));
import matlab.unittest.TestSuite
import matlab.unittest.TestRunner

switch scope
    case 'fast'
        suite = [TestSuite.fromClass(?TestCv), TestSuite.fromClass(?TestModels)];
    otherwise
        suite = TestSuite.fromFolder(fullfile(root, 'tests'));
end
runner = TestRunner.withTextOutput('Verbosity', 2);
results = runner.run(suite);
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('%d passed, %d failed, %d incomplete  (%.1f s)\n', nnz([results.Passed]), ...
    nnz([results.Failed]), nnz([results.Incomplete]), sum([results.Duration]));
if nargout == 0, clear results; end
end
