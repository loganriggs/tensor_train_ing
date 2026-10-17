# Fixed lagged scale and interpretation scope

At a frozen checkpoint, LaggedRMS computes `N(x)=g*x/sqrt(s+eps)`. For positive c and valid new statistic, the transformation `g'=c*g`, `s'=c²(s+eps)-eps` leaves N, its Jacobian, and the whole model unchanged. This is an exact algebraic identity including epsilon, not an asymptotic RMS scaling claim.

Current `analysis.gain` exports g but omits the fixed divisor. Accordingly `_readers` and `mlp_local_tensor(..., fold_gain=True)` change under this equivalent checkpoint representation. A float64 tiny-model reproduction with s=4 and c=3 gives:

- Maximum logit change: 1.39e-17; branch change: 1.67e-16.
- Exported reader norm ratio: 3; exported local tensor norm ratio: 9.
- Original exported tensor applied directly to x has relative error 3.000001 against MLP(N(x)), because the omitted squared divisor is 4.000001.
- Independent reconstruction using effective gain g/sqrt(s+eps) has maximum error 1.53e-16.

These results do not imply the current gain-folded numerator tensor is algebraically wrong under a numerator-only definition. They show it must not be presented as the effective fixed-checkpoint polynomial, or as invariant across this function-preserving norm reparameterization. Positive scalar changes do not alter cosine edges or scale-normalized rank metrics; raw edge magnitudes and coefficient eigenvalues do change.

Claude request: for fixed LaggedRMS sites, either fold the constant divisor into effective readers/local forms, or export the actual fixed divisor and explicitly label numerator-only semantics at each site. Separate this from dynamic RMS denominators, which cannot simply be folded into constant coefficients. Add a regression applying the exact epsilon-aware gauge above and comparing effective tensors/readers to the unchanged forward function. Output/final norm factors and residual/logit scalars still require their own treatment. No optimizer-equivalence claim is made: this is a frozen-checkpoint identity.

Reproduce with `CUDA_VISIBLE_DEVICES='' /venv/main/bin/python runs/codex_lagged_gauge_audit/reproduce.py`. Evidence includes current source hashes. No production edits, training, real corpus or final-test probes.
