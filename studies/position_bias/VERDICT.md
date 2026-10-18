# Fixed bias versus independent jitter

Condensed English scientific translation. Eight paired seeds 505010000–505010007; verdicts concern means in the frozen model. All three numerical predictions were reported supported, without proving a universal phase transition. SPEC and training code were not original.

Bias .383 lambda: accuracy 50.3235%,95% interval[47.1573,53.4624], decline 48.1857pp from zero bias. Gaussian jitter sigma .383/.45:98.4222/98.3795%; .383 jitter exceeds equal fixed bias by 48.0988pp. Polar correction with 64 pilot pairs:98.4863%, interval[98.3887,98.5870], improvement 48.1628pp. Every seed reportedly meets the original accuracy/gain thresholds, but the original threshold specification is unavailable and is not reconstructed as a new preregistration.

Full condition means: fixed 43.2449%,CSI 98.5916%; bias 0/.15/.383/.5 gives 98.5092/98.5138/50.3235/43.2449%; jitter 0/.15/.383/.45 gives 98.5092/98.5733/98.4222/98.3795%. Structural reuse at bias .5=fixed and jitter 0=bias 0 does not add independent runs .72 new trainings including 16 prechecks,88 reported rows. Precheck excess CE .743492608798 passes .10.

E[J0(2 pi e)] = exp(-pi^2 sigma^2) I0(pi^2 sigma^2)>0 at finite Gaussian sigma; no sigma-independent positive lower bound. Deterministic first zero .382739874781 lambda; .383 lies slightly beyond, J0=-.000848213848. Gaussian expectations .375396053794/.308639411414 at .383/.45. Full C10 does not vanish at .383: off-diagonal Frobenius norm 1.17908945 and eigenvalues[-.45512685,-.38139010,-.17106372,1.00418781]. Polar therefore does not create information from an independent return channel.

Mean gradient cosines for zero bias,critical bias,jitter .383,jitter .45,polar: .883210,.012643,.470658,.396632,.745439; negative-inner-product batch percentages 0,47.9167,8.4375,12.7083,1.1458. Positive scalar expectation is not a per-step descent guarantee.

Scope:4Rx/2Tx isotropic complex Gaussian fields, perpendicular motion, binary task with three binary features, independent exchanges. Bias shared across the array; jitter independent by element/exchange. CSI has 2666 examples/84 updates versus 3840/120 for physical arms and includes 16-bit gradient quantization. Polar also changes normalization and feedback, so its gain cannot be assigned solely to polar decomposition. No actuator limits, acceleration, energy or measured hardware. Historical audit reports zero endpoint/replay error,1.334e-10 finite-difference error and no collision in 148 registered streams; raw artifacts needed to repeat it are absent.
