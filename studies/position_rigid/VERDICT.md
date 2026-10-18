# Rigid Gaussian jitter

Condensed English scientific translation. Eight paired seeds 555510000–555510007 and fixed 120 updates. SPEC and training code were not original. Claims 1/2/3 received finite-configuration numerical support; no universal immunity is proved.

Rigid sigma .383/.45 lambda:98.4909/98.4192% accuracy; Gaussian expected correlations .375396/.308639. Rigid minus independent jitter mean accuracy differences at sigma .15/.383/.45: -.0931,+.0198,-.0153pp; CE differences .000456,.000659,.002959. Frozen mean thresholds: each sigma within 1pp and .01CE. Across-sigma mean absolute difference .0427pp/.001358CE. At .45 one seed has CE difference .016276, exceeding .01: the mean rule is not a per-seed guarantee.

At .383, rigid 98.4909% versus fixed bias 53.2944%, improvement 45.1965pp,95% interval[40.5228,48.1445]. Per-seed gains[29.8584,49.0112]pp include one below 30pp. Fixed_S0 mean 48.1262% (range 37.6831–51.0010), CSI 98.5229%; rigid 0/.15/.383/.45 means 98.5046/98.4558/98.4909/98.4192%; independent counterparts 98.5046/98.5489/98.4711/98.4344%. Zero-jitter structural reuse does not add independent samples.

E[J0(2 pi e)] = exp(-pi^2 sigma^2) I0(pi^2 sigma^2)>0 for finite sigma, approaching zero without a uniform positive lower bound. This scalar identity holds for both rigid and independent errors sharing the same Gaussian marginals. Rigid motion preserves same-time covariance C11=C00, but not forward/return equality. Independent jitter changes C11 and conditional innovation even when expected C10 agrees. Full 4x4 joint covariance was used.

Probability of a negative instantaneous J0 at sigma 0/.15/.383/.45:0/1.0723/29.6163/34.6308%. Mean positivity does not ensure each exchange is positive, nor does scalar sign describe an entire complex channel. Bias .383 is slightly beyond the first zero .382739874781; cross-channel off-diagonals remain nonzero.

Independent precheck CE gap .646595353 passes .10.96 new trainings and 104 rows including zero-jitter reuse were reported. Historical audit maximum CE error 1.39e-17, accuracy error 0, gradient reconstruction 4.45e-16. A geometry description incorrectly said independent errors for rigid arms; actual rigid errors were identical across elements and C11=C00. No continuous trajectory, correlated time error, finite stroke, acceleration, coupling, energy or deployment delay was modeled. Four receive/two transmit binary model, perpendicular motion, eight means and constant 16 return restoration limit external validity.
