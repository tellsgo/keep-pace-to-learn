# Residual-displacement results

K1: finite-grid operational support, without locating the continuous low-speed boundary. Observed endpoints 0,.8,.9 versus theory .096051,.728815,.909605; errors .096051,.071185,.009605. At 3 km/h all eight moving cells exceed 80%, and only fixed fails (49.0067% versus 85.5209% at kappa .5). At 10 km/h the bracket [.8,.85] does not contain the theoretical .728815: positive correlation does not ensure 80% accuracy in 120 updates.

K2: finite-configuration support. Main 24-cell Spearman .952692958, seed-bootstrap descriptive 95% interval [.841857,.970781]. Including fixed gives .952410646, including CSI .958158586. Within-speed correlations .928571,.928571,.857143. These are associations, not a sufficient causal law.

K3: descriptive comparison only. Second lobe (30,.7): Re(phi)=.462681, Im(phi)=-.151548, abs(phi)=.486868, accuracy 98.3490%, CE .075759. Matched first lobe (10,.8): .374745,-.331621,.500406, accuracy 67.6636%, CE .603098. Paired advantage 30.6854pp,95% interval[26.8737,35.0220]. Fixed second-lobe supplemental point (10,0) gives 84.7275%, paired advantage 17.0639pp [12.7594,22.3303].

Independent precheck CE gap .638748129 passed .10. All 256 prescribed precheck/formal trainings were reported completed without seed replacement. Full cell means, uncertainty and endpoints are retained in results/cells.csv and results/claims.json. Historical audit claims are not fresh verification in this release; missing raw worlds prevent replay. Scope is one frozen CDL-A orientation, narrowband independent drops, constant 16 return scaling and 120 updates, without actuator limits, continuous trajectories or deployment costs. Scalar zero crossings and finite-budget learning thresholds are distinct.
