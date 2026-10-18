# Matched benchmark and threshold refinement: frozen protocol

Condensed English scientific translation. No training-grid or seed changes after results; full eight-seed means are the unit of decision, paired across conditions;20000 seed bootstrap resamples provide descriptive 95% intervals.

B1: CDL-A/C,3/30 km/h,40 ms,3.5 GHz,4Rx/2Tx half-wavelength y-axis arrays, x-axis motion, scalar vertical polarization, static scatterers .23/24 clusters x20 rays,300 ns delay spread,8 tones (k-3.5)MHz. Forward/return share phases within each independent exchange. Seeds 2160101000–2160101007, paired geometry, data, noise and initial state. Average tones before seeds.

Five arms: uncorrected S0; exact MA kappa 1; polar correction from 64 pilot pairs with quantized unitary matrix; wait until the first accessible negative-correlation local minimum after 40 ms and return with minus identity; exact-CSI gradient oracle without reverse noise. Waiting searches [d0,d0+3] on 3001 points, records absence without enlarging range. Physical arms normalize each sample and restore its norm using 16-bit feedback on[0,16].

Each arm/tone uses 3840 samples,batch 32,120 updates,8192 test samples. Three independent binary inputs,label1[x0*x1>0],complex 2x3 encoder,tanh(abs(y)^2)+logistic. Adam .025/.06,beta(.9,.99),epsilon 1e-8; encoder initial norm .4 and limit 1/sqrt(3); receiver initial .8/-.4,parameter clipping 4,L2 .001,noise std .1/.03. This matches learning budgets, not total RF, feedback or wall time.

B1 criterion: signed excess CE(method)-CE(CSI) for MA and polar across four conditions (eight comparisons). All at most .02 supports; any greater than .05 falsifies; otherwise prediction not met, not falsified. Also report absolute differences. Accuracy and wait are descriptive.

Per seed/condition:30720 forward and 30720 return tone-vector uses; parallel 8-tone interpretation has 3840 exchanges/7680 RF vector slots. Polar adds 2048 pilot tone uses/256 wideband pilot vector slots. Norm feedback 491520 bits; polar adds 131072 observation bits and 512 matrix bits. Test 65536 forward tone uses separately. CSI acquisition is unpriced. MA displacement .033333/.333333 m per exchange and 128/1280 m accumulated, with reset travel additional; do not multiply by tones or array elements. No total wideband power, actuator energy or timing guarantee.

B2: same residual-study constant 16 return scaling (variance .2304), not B1 norm feedback. CDL-A,3 km/h,40 ms,narrowband,100 ns delay parameter; same 3840/32/120/8192 and learner. Kappa {0,.05,.1,.15,.2,.3,.4}, seeds 2160102000–2160102007 paired across kappa. Philox roles separated. Average geometric phi across eight seeds; search[0,4] on 40001 points plus brentq before training. First root .351797755345; kappa_star .096003024625.

B2 criterion: scan kappa .4 down to 0, first mean below 80% is the operational endpoint. Error at most .1 supports, greater than .2 falsifies, otherwise unresolved. Report a bracket only if a successful adjacent point exists; all successful means left censoring. Highest point already failing cannot establish a continuous threshold. Keep all cells for descriptive coverage after any falsification. No interpolation, retries or post-result changes. Original training source and raw worlds were not original in this release input.
