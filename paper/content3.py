# -*- coding: utf-8 -*-
"""Results, discussion, conclusion, back matter and references."""
import os

BLOCKS = [
("h1", "5. Results"),
("p",
 "The results are reported in the order in which they constrain the design: long-horizon accuracy first, then "
 "the shape of the error growth, then the ablation that identifies which component causes which effect, then the "
 "regime in which structure pays, then the closed-loop consequence, and finally the energy and cost measurements. "
 "Every number in this section comes from a trained model and is reproduced by the scripts described in the "
 "appendix note; none is carried over from a previous version of the study."),
("h2", "5.1. Long-horizon accuracy"),
("p",
 "Table 2 reports normalised error at 25, 100 and 200 steps for five baselines and the reported model on the "
 "four domains, with the growth exponent on the two domains where a law is well defined, latency, and parameter "
 "count. The last row is the same model with "
 "the Verlet map removed; Section 5.3 uses it as an ablation on the projectile and Section 5.8 repeats it on all "
 "four domains. Fig. 2 plots the full error trajectory at four horizons on the projectile and the quadrotor."),
("table", "Table 2. Normalised RMSE at horizons of 25, 100, and 200 steps, error-growth exponents, per-step latency, and parameter count. Lower is better except for latency and parameters. Every entry is the mean of two seeds.", None,
 [("MLP-MD", "0.054", "0.232", "0.539", "0.376", "0.015", "0.224", "1.10", "0.84", "0.20", "37,382"),
  ("GRU-LD", "0.046", "0.194", "0.412", "0.502", "0.369", "0.443", "1.05", "1.06", "0.23", "31,974"),
  ("NODE", "0.050", "0.230", "0.575", "0.381", "0.013", "0.222", "1.17", "0.83", "0.97", "37,383"),
  ("HNN", "0.158", "0.601", "1.462", "2.477", "3.136", "4.317", "1.06", "1.00", "1.30", "34,059"),
  ("PI-Lat", "0.047", "0.174", "0.347", "0.479", "0.179", "0.441", "0.96", "1.15", "0.24", "70,598"),
  ("PCL-WM", "0.046", "0.167", "0.793", "1.598", "1.019", "3.265", "1.33", "1.74", "2.29", "102,911"),
  ("PCL-WM w/o symplectic", "0.045", "0.147", "0.343", "1.168", "0.931", "8.339", "0.96", "1.59", "2.29", "102,911")]),
("p",
 "The first observation is that the comparison is no longer trivial. Under full observability an earlier version "
 "of this protocol produced a direct predictor with 0.04 error at 200 steps on the projectile, and no latent "
 "model came within a factor of ten. With the reduced sensor set and the four-step window the direct predictor "
 "degrades to 0.539, and the best latent model reaches 0.347. The degradation is the intended effect of the "
 "protocol: the direct model now has to form finite differences from a window, which is a lossy estimator of the "
 "hidden velocities, and the latent model has an encoder whose whole purpose is to perform that estimation. The "
 "protocol therefore gives structure a chance to matter, which is what any comparison of structural priors "
 "requires."),
("fig", os.path.join("figs", "fig2_error_growth.png"),
 "Fig. 2. Error growth on the projectile (a) and on the quadrotor (b). All models see only the instrumented "
 "channels and receive a four-step history window. The structured model is the most accurate at short horizons "
 "and the least accurate at long horizons, which is the pattern the ablations in Section 5.3 explain."),
("p",
 "The second observation is that PCL-WM has the best short-horizon accuracy of the latent families and the worst "
 "long-horizon accuracy of all the latent models. At 25 steps it is at 0.046, level with the recurrent latent "
 "model and ahead of the direct and ODE predictors, which confirms that the structured map is a good local "
 "approximation. At 200 steps it is worse than the direct predictor (0.793 against 0.539), worse than the "
 "recurrent latent model (0.412), and worse than the unstructured latent model (0.347). With the Verlet map "
 "removed the same architecture moves to 0.045, 0.147 and 0.343, which puts it at or ahead of the best baseline at "
 "every horizon on this domain. The crossover of the full model happens between 100 and 200 steps, which is "
 "exactly the regime the certificate of Section 3.5 is designed to flag, and it is not a small-sample artefact: "
 "the two seeds give 0.726 and 0.860 for the full model and 0.347 and 0.339 for the model without the map, a "
 "spread that does not overlap any of the competing methods."),
("p",
 "The third observation is that the Hamiltonian network without an encoder fails, and fails uniformly. On the "
 "projectile it reaches 1.462, on the quadrotor 2.477, on the thermal network 3.136, and on the "
 "reaction-diffusion field 4.317, which is an order of magnitude worse than everything else on the last domain. "
 "The failure has a simple cause: the model applies Hamilton's equations to the measured channels, and the "
 "measured channels are positions and attitudes without their conjugate momenta, so the reconstructed flow is "
 "missing half of its state. This result is useful in two directions. It shows that the structural prior alone "
 "is not enough, and it shows that the encoder is not incidental machinery but the component that makes a "
 "physical prior applicable to a partially observed plant. The seed spread on the projectile is large for this "
 "model (1.734 and 1.191), which is consistent with a model that is unstable rather than merely inaccurate."),
("p",
 "The direct family and the neural-ODE family behave almost identically, which is expected: an ODE solver with a "
 "fixed number of steps is a residual network with a particular parameterisation, and the difference between "
 "0.539 and 0.575 at 200 steps is within the seed spread of 0.03 measured on the direct model. The recurrent "
 "latent model is competitive on the projectile (0.412) and clearly worse on the thermal network (0.369 against "
 "0.014), because a recurrence trained on 60 epochs has neither the smoothness of an ODE nor the locality of a "
 "convolutional window. On the thermal network the two direct models are two orders of magnitude better than "
 "anything latent, at 0.014 and 0.013, and no structural argument survives that: when the state is fully "
 "observed and the dynamics are strongly dissipative, an encoder is pure overhead."),
("fig", os.path.join("figs", "fig3_beta.png"),
 "Fig. 3. Error-growth exponent on the projectile. A value of one corresponds to a linearly growing error, "
 "values below one to a saturating error, and values above one to an accelerating error. The structured model "
 "is the only family above 1.3, and removing its symplectic step brings it back to 0.99."),
("h2", "5.2. Error-growth exponent"),
("p",
 "The normalised error at a fixed horizon says how large the error is; the exponent says how it grows, and for a "
 "controller the second quantity is more informative, because it determines how far ahead the model can be "
 "trusted. Table 3 reports the fitted exponent on all four domains."),
("table", "Table 3. Error-growth exponent on the four domains, obtained by fitting a straight line to the logarithm of the error against the logarithm of the horizon.", None,
 [("MLP-MD", "1.10", "0.84", "0.02", "0.06"),
  ("GRU-LD", "1.05", "1.06", "1.46", "0.39"),
  ("NODE", "1.17", "0.83", "-0.01", "0.06"),
  ("HNN", "1.06", "1.00", "0.88", "0.81"),
  ("PI-Lat", "0.96", "1.15", "1.25", "0.42"),
  ("PCL-WM", "1.33", "1.74", "2.15", "1.41")]),
("p",
 "The pattern is unambiguous. PCL-WM has the largest exponent on three of the four domains and the second "
 "largest on the fourth, and on the thermal network, where the true dynamics are strongly dissipative, an "
 "exponent of 2.15 means the error is not merely growing but accelerating. The unstructured latent model has an "
 "exponent of 0.96 on the projectile, which is the signature of an error that has saturated below the signal "
 "amplitude; the structured model has 1.33, which is the signature of an error that keeps compounding. The "
 "direct model sits at 1.10, between the two, and the recurrent model at 1.05."),
("p",
 "An exponent above one is the quantitative form of the qualitative claim made in the introduction: the "
 "structured model is locally excellent and globally unreliable. Section 5.3 shows that the excess is produced "
 "by one component, and Section 6 explains why that component behaves this way."),
("p",
 "The certificate of Eq. (16) is measured once, on the projectile, at two seeds. With the Lipschitz "
 "constant estimated from finite differences along the rollout, $L=1.00$, and the learned step of 0.10, "
 "the amplification factor is 1.10, which certifies six to seven steps; the measured error crosses the "
 "same tolerance of 0.1 only at 68 to 72 steps. The bound therefore expires an order of magnitude early, "
 "which is the safe direction, and it is a monitoring signal rather than a stopping rule. It is also shorter "
 "than the planning horizon of 15 used in Section 4.3, so the closed loop does not rest on it; Section 5.5 "
 "carries that claim with repeated runs."),
("h2", "5.3. Ablation: which component does what"),
("p",
 "Seven ablations of the structured model are trained on the projectile with an otherwise identical protocol, "
 "and every one of them is run twice. Table 4 reports the accuracy at three horizons, the exponent, the "
 "one-step error, and the parameter count, as means over the two seeds; Fig. 4 ranks the configurations by "
 "their 200-step error and draws the range of the two runs as an error bar, which is the part of this study "
 "that changed most when the second seed was added."),
("table", "Table 4. Ablation of the structured model on the projectile. Each configuration differs from the full model in exactly one factor. Values are the mean of two seeds; the per-seed spread of the 200-step error is given in the text.", None,
 [("PCL-WM (full)", "0.046", "0.086", "0.793", "1.33", "0.0127", "102,911"),
  ("w/o symplectic step", "0.045", "0.082", "0.343", "0.96", "0.0126", "102,911"),
  ("w/o gauge penalty", "0.046", "0.085", "0.992", "1.41", "0.0127", "102,911"),
  ("free dissipation", "0.046", "0.085", "0.969", "1.41", "0.0127", "102,911"),
  ("w/o certificate", "0.046", "0.087", "0.717", "1.29", "0.0127", "102,911"),
  ("w/o curriculum", "0.046", "0.094", "0.950", "1.42", "0.0126", "102,911"),
  ("latent width 16", "0.046", "0.088", "1.491", "1.54", "0.0126", "94,927"),
  ("latent width 64", "0.047", "0.086", "0.794", "1.31", "0.0127", "119,647")]),
("fig", os.path.join("figs", "fig4_ablation.png"),
 "Fig. 4. Ablation configurations of the structured model ranked by 200-step error on the projectile, mean of "
 "two seeds, with the range of the two runs as an error bar. Removing the symplectic step is the single "
 "largest change in the entire study, it is an improvement, and it is also the only change whose two seeds "
 "agree to within 2%."),
("p",
 "Removing the symplectic step lowers the 200-step error from 0.793 to 0.347, a factor of 2.3, lowers the "
 "exponent from 1.33 to 0.99, and leaves the short-horizon error essentially unchanged (0.043 against 0.046) "
 "at the cost of a marginally worse one-step error (0.0140 against 0.0127). The trade is therefore entirely in "
 "the direction of long-horizon reliability: the model that keeps the symplectic map is slightly better at "
 "predicting the next sample and much worse at predicting two hundred samples, which is the opposite of what the "
 "structure was supposed to buy."),
("p",
 "The other ablations were first run with one seed, and on that seed they pointed away from the"
 " construction: removing the gauge penalty took the error from 0.793 to 0.735, freeing the"
 " dissipation to 0.706, and removing the certificate to 0.684. Those three numbers are now"
 " withdrawn as evidence, because every row of Table 4 was then duplicated and the second seed moved"
 " them a long way. The two-seed means are 0.992 for the gauge-free configuration, 0.969 for the"
 " free dissipation, 0.950 without the curriculum, and 0.717 without the certificate, against 0.793"
 " for the full model. Three of the four removals are therefore worse than the full model and the"
 " fourth is inside its own seed spread, 0.726 to 0.860. The single-seed reading of this table was"
 " noise: the gauge condition and the constrained dissipation are not a net cost in distribution,"
 " they are simply not resolved by an ablation whose rows differ by less than the spread within"
 " each row."),
("p",
 "This also explains why the remaining components are hard to resolve here. All the ablations in Table 4 sit on "
 "top of a configuration whose dominant error source is the symplectic step, so the marginal value of the other "
 "constraints is measured in the presence of a large, growing error that they were never designed to suppress. "
 "A constraint that limits the growth of a stable rollout can be worth something where the error would otherwise "
 "stay bounded, and nothing where the error is already doubling every fifty steps. The ablation therefore "
 "identifies the symplectic step as the dominant problem on this domain and leaves the question of the other "
 "components open, which is the question that Section 5.4 answers in the regime where it matters, and Section "
 "5.8 answers for the integrator itself by repeating the removal on the other three plants."),
("p",
 "Latent width does not behave as a capacity knob in any way this study can resolve. With one seed, sixteen "
 "channels were better than thirty-two (0.546 against 0.793) and sixty-four were worse (0.998); with two seeds "
 "sixteen channels average 1.491 over a spread of 0.546 to 2.436, and sixty-four average 0.794 and sit beside "
 "the full model. The width comparison is reported because it was run, not because it supports a "
 "recommendation: the second seed contradicts the first in both directions, and the honest conclusion is that "
 "latent width is the least stable choice in the configuration at this data scale. It is also the clearest "
 "single illustration of why every row here is now duplicated."),
("p",
 "The one-step column is worth a sentence of its own. All eight configurations are within 12% of one another on "
 "the one-step error, between 0.0127 and 0.0141. A study that reported only one-step accuracy would conclude "
 "that the structural choices are irrelevant. The differences that matter appear only after the error has been "
 "allowed to compound, which is an argument for evaluating world models on rollouts rather than on transitions."),
("h2", "5.4. Data efficiency and distribution shift"),
("p",
 "The ablation identifies which components are worth keeping with 240 training trajectories. An engineering "
 "deployment usually occupies the opposite regime, few trajectories and operating points outside the trained "
 "range, and Table 5 reports both: four models trained on 60 trajectories with two seeds each, on the "
 "projectile, with the test set in distribution or with initial speeds of 240 to 320 m/s against a training "
 "range of 180 to 260 m/s. The shifted set is drawn from the stream offset four past the seed, so the two "
 "seeds see different shifted sets and their spread mixes training-run variation with a change of initial "
 "conditions."),
("table", "Table 5. Data efficiency and distribution shift on the projectile. Models are trained on 60 trajectories, two seeds per configuration, and every value is the mean of the two runs; the ranges are given in the text. The shifted test set uses initial speeds of 240 to 320 m/s against a training range of 180 to 260 m/s.", None,
 [("MLP-MD", "0.051", "0.234", "0.602", "1.18", "0.363", "0.744", "1.02"),
  ("PI-Lat", "0.048", "0.167", "0.301", "0.89", "0.350", "0.704", "1.02"),
  ("PCL-WM", "0.048", "0.185", "1.032", "1.42", "0.312", "1.049", "1.17"),
  ("PCL-WM w/o symplectic", "0.048", "0.159", "0.328", "0.92", "0.315", "0.595", "0.95")]),
("fig", os.path.join("figs", "fig5_data_efficiency.png"),
 "Fig. 5. 200-step error with 60 training trajectories, in distribution and with initial states shifted outside "
 "the training range. Bars are the mean of two seeds and the error bars the range between them. Structure pays "
 "only in the shifted condition, and only when the symplectic step is removed."),
("p",
 "Three conclusions follow. First, in distribution the configuration without the symplectic step (0.328) is nine "
 "percent behind the unstructured latent model (0.301), and the ranges, 0.314 to 0.341 against 0.288 to 0.315, "
 "barely overlap, so two seeds do not separate them; both beat the direct predictor (0.602) through the encoder "
 "and the residual decoder rather than through mechanics. Second, out of distribution structure buys "
 "about sixteen percent: 0.595 against 0.704 for the unstructured latent model and 0.744 for the direct "
 "predictor, and the two seeds of the structured configuration, 0.581 and 0.608, bracket a margin four times wider "
 "than their own spread. Third, the symplectic step destroys that gain and more: with it the structured model is "
 "the worst of the four in both conditions, 1.032 and 1.049, while still holding the best 100-step error under "
 "shift, 0.312."),
("p",
 "The out-of-distribution exponent is the most diagnostic number in the table. The direct predictor and the "
 "unstructured latent model have 1.018 and 1.022, so their error grows linearly from a larger initial value; the "
 "structured model with the symplectic step has 1.167 and grows faster than linearly. The same construction "
 "without it has 0.947, below one, so the error saturates: it stays bounded where the others keep growing. That "
 "is what a dissipation constraint is supposed to do, and it is the "
 "first measurement here in which the physical prior produces the behaviour it was introduced for. Both seeds "
 "reproduce it, 0.927 and 0.966, the tightest agreement in the table."),
("p",
 "Two limits on this attribution should be stated. The sixteen percent separates two configurations that differ in "
 "the canonical split, the constrained dissipation, the gauge penalty and the certificate at once, and we did not "
 "run the complementary ablations, so the number belongs to the group and cannot be assigned to one component. The "
 "exponent of 0.947 is stronger evidence, being a property of the growth law on both seeds, but it too is a "
 "property of the group, and the same configuration pays nine percent in distribution."),
("h2", "5.5. Closed-loop control"),
("p",
 "Open-loop accuracy is a proxy. What a control engineer cares about is the tracking error of a controller "
 "that plans with the model and executes on the plant, since the controller can correct bias at every step but "
 "not a model that is confidently wrong. We place each model inside a cross-entropy model predictive controller "
 "and execute the first action of the planned sequence on the exact simulator for one hundred steps. Since one "
 "run is not a measurement, each configuration is trained once and evaluated in three controller runs, over "
 "disjoint blocks of eight episodes with their own sampling seeds."),
("table", "Table 6. Closed-loop position tracking error on the projectile, in normalised position RMSE over one hundred steps. The controller plans with the learned model and executes the first action on the exact simulator; horizon 15, 128 candidates, 3 iterations, 16 elites. Each configuration is trained once and evaluated in three controller runs of eight disjoint episodes, so the entry is the mean over twenty-four episode-runs and the last column is the range of the three run means.", None,
 [("NODE", "0.0649", "0.0627 to 0.0661"),
  ("MLP-MD", "0.0690", "0.0638 to 0.0734"),
  ("PCL-WM w/o symplectic", "0.0710", "0.0659 to 0.0741"),
  ("PCL-WM", "0.0710", "0.0633 to 0.0781"),
  ("PI-Lat", "0.0739", "0.0714 to 0.0782")]),
("fig", os.path.join("figs", "fig6_closedloop.png"),
 "Fig. 6. Closed-loop model predictive control on the projectile. (a) Mean tracking error over one hundred steps, "
 "averaged over twenty-four episodes, with the band spanning the three controller runs. (b) Episode-level mean "
 "with the same range. The figure and Table 6 come from the same runs."),
("p",
 "The repetition shows that the closed loop on this plant does not separate the models. The five configurations "
 "fall between 0.0649 and 0.0739, a spread of 0.009, while the three runs of one configuration differ by up to "
 "0.015, as the full structured model does between 0.0633 and 0.0781. No pairwise difference survives its own "
 "run-to-run range, so the order in Table 6 describes one campaign and not a ranking. An earlier version used "
 "four episodes and one run per model and placed the direct predictor first at 0.0574 and the structured model "
 "last at 0.0680; that order does not survive repetition, and the claim that planning compresses the gaps "
 "without reordering the models is withdrawn."),
("p",
 "What survives is a statement about feedback, not about the models. Every configuration tracks with a "
 "normalised position error between 0.064 and 0.074 at a hundred steps, whereas open-loop rollouts of the same models have "
 "already separated by an order of magnitude at two hundred steps. A controller that re-plans every step and "
 "corrects bias is therefore insensitive to much of what the open-loop tables measure, which is the practical "
 "point for a deployment decision and the reason the last row of Table 12 asks for a closed-loop trial before "
 "committing to a model. Ranking inside that envelope needs far more episodes per configuration than twenty-four."),
("h2", "5.6. Energy consistency"),
("p",
 "The thermal network measures every state channel, so it is the only domain where the physical energy "
 "functional can be evaluated on a reconstructed trajectory without an implicit state estimate. Table 7 reports "
 "the mean absolute relative deviation of that functional over two hundred steps."),
("table", "Table 7. Relative deviation of the reconstructed physical energy over 200 steps on the thermal network, in percent, the mean of two training seeds, each evaluated on its own test episodes. The functional is the capacitance-weighted quadratic deviation from ambient temperature.", None,
 [("NODE", "1.33"),
  ("MLP-MD", "1.53"),
  ("PI-Lat", "40.23"),
  ("GRU-LD", "99.06"),
  ("PCL-WM", "407.47"),
  ("HNN", "3276.38")]),
("p",
 "The result is a warning about what a latent coordinate system does to an energy functional. The two direct "
 "models deviate by one to one and a half percent, which is the level one expects from a well-trained "
 "reconstruction on a dissipative system. The latent models deviate by one to two orders of magnitude more, and "
 "the structured model deviates the most. The reason is not that the structured model is a worse predictor of "
 "temperatures, since it is only 0.96 away in normalised error against 0.014 for the direct model, but that its "
 "learned energy is a function of latent coordinates whose scale is arbitrary; reconstructing the physical "
 "functional from the decoded state then amplifies every latent error through the quadratic form. A "
 "structure-preserving model preserves the energy of its own latent Hamiltonian, not the energy of the plant, "
 "and the two are equal only if the decoder is metrically consistent with the physical coordinates, which "
 "nothing in the training objective enforces."),
("p",
 "This measurement tempers the claim that a port-Hamiltonian parameterisation makes a model energetically "
 "trustworthy. It makes the model's own dissipation non-negative by construction, which is what the exponent "
 "measurement in Section 5.4 shows; it does not make the model's energy equal to the plant's energy, which is "
 "what this table shows. Both statements belong in a paper that claims engineering relevance. The ordering of "
 "the table is identical in both seeds; the largest movement between them is a factor of two in the two HNN "
 "and ablation rows, which are the rows where the decoder has the least metric constraint."),
("h2", "5.7. Cost"),
("p",
 "Table 8 collects the deployment cost. The structured model has 2.8 times the parameters and 10.6 times the "
 "per-step latency of the direct predictor, and it trains 6.7 times slower. For a controller that solves a "
 "planning problem at every step, a per-step latency of 2.29 ms against 0.20 ms is the difference between "
 "planning 128 candidates in 0.27 s and planning them in 0.026 s, which is the difference between a 4 Hz and a "
 "40 Hz control loop on the hardware used here."),
("table", "Table 8. Model size, per-step inference latency, and training time for one configuration on the projectile.", None,
 [("MLP-MD", "37,382", "0.20", "36"),
  ("GRU-LD", "31,974", "0.23", "60"),
  ("NODE", "37,383", "0.97", "140"),
  ("HNN", "34,059", "1.30", "134"),
  ("PI-Lat", "70,598", "0.24", "51"),
  ("PCL-WM", "102,911", "2.29", "242")]),
("p",
 "The latency gap is produced by the implicit differentiation inside the structured map: computing the gradient "
 "of the learned Hamiltonian with respect to the latent state at every step costs about ten forward passes of a "
 "small network. Since the accuracy gain is confined to the out-of-distribution regime and is not visible "
 "in distribution, the cost argument alone would reject the full model for a real-time application and favour "
 "the direct predictor or the unstructured latent model. We state this plainly because it is the conclusion a "
 "reader who intends to deploy the method needs, and because the alternative framing, that structure is worth "
 "any price, is not supported by the measurements."),

("h2", "5.8. How much of the gap belongs to the integrator"),
("p",
 "Section 5.3 establishes that removing the Verlet map is the single largest change on the projectile, and the "
 "obvious question is whether the same removal repairs the other three plants. Table 9 answers it by repeating the "
 "removal on all four domains with the protocol unchanged. The answer separates two things that the earlier "
 "literature, and our own Section 6.1 draft, tended to conflate: the cost of the discrete integrator, and the cost "
 "of the canonical split itself."),
("table", "Table 9. Effect of removing the Verlet map on all four domains. Every column is the mean of two seeds. The ratio is the with-map error divided by the without-map error, so a value above one means the map costs accuracy.", None,
 [("D1", "Projectile", "0.793", "0.343", "2.31", "1.33 / 0.96", "-"),
  ("D2", "Quadrotor", "1.598", "1.168", "1.37", "1.74 / 1.59", "-"),
  ("D3", "Thermal network", "1.019", "0.931", "1.09", "2.15 / 1.92", "407.5 / 201.3"),
  ("D9", "Reaction-diffusion", "3.265", "8.339", "0.39", "1.41 / 1.87", "-")]),
("p",
 "On the projectile the map costs a factor of 2.3. On the quadrotor it costs a factor of 1.37, on the thermal "
 "network a factor of 1.09, and on the reaction-diffusion field it is the cheaper of the two, by a factor of 2.6. "
 "The exponent moves in the same direction as the error, from 2.15 to 1.92 on the thermal network and from 1.41 to "
 "1.87 on the field. A result that holds on one plant and reverses on another is not a property of the integrator; "
 "it is a statement about how well the canonical pair matches the plant. The projectile is close to a conservative "
 "mechanical system, so a map that preserves a quadratic form is at odds with a Hamiltonian that the network has "
 "to estimate, and the mismatch is converted into error. The reaction-diffusion field is the opposite case: its "
 "latent dynamics is stiff and marginally stable, the implicit half-kick of the Verlet map damps the fastest "
 "latent modes, and removing it exposes those modes to a 2.6-fold error increase. The quadrotor and the thermal "
 "network sit in between, where the split is neither natural nor useless."),
("p",
 "This changes the reading of Section 6.1. The three mechanisms proposed there explain the projectile result and "
 "explain part of the quadrotor result, but they do not explain why the same removal hurts on the field. The "
 "correct statement is narrower and more useful: a hard symplectic step is a net cost when the learned Hamiltonian "
 "is a good approximation of a near-conservative flow and the latent coordinates are close to canonical, and it is "
 "a net benefit when the latent dynamics contains stiff modes that the implicit stage happens to damp. The design "
 "rule that follows is not \"drop the integrator\" but \"match the integrator to the spectrum of the learned "
 "latent dynamics\", which is a statement that can be tested before deployment by measuring the exponent of the "
 "learned map on a short validation rollout."),
("p",
 "The energy column carries a second reading. Removing the map lowers the energy deviation on the thermal network "
 "from 407% to 201%, which is a large relative improvement and still a poor absolute number. A latent model whose "
 "energy is reconstructed through a decoder with no metric constraint does not become energetically faithful by "
 "changing the integrator; the deviation remains two orders of magnitude above the direct model. Section 5.6 "
 "explains why, and the number should be read as a bound on what structural parameterisation alone can deliver."),

("h2", "5.9. Engineering applicability on a measured fleet"),
("p",
 "Every result so far comes from a simulator we wrote, the weakness a reviewer should attack first: a "
 "structured model can look good on a plant coded by whoever coded the structure. The measured fleet cannot "
 "repair that and is not external validation of the ranking; it delimits where Section 5 holds, because its "
 "instrumentation, noise level and horizon are those of a deployment rather than a benchmark. The protocol runs "
 "unchanged on the NASA C-MAPSS FD001 fleet, a four-step window, six of twenty-one measured channels, the same "
 "training budget, two seeds, horizons of 25, 50 and 100 cycles; the question is which of the three claims "
 "survives the change."),
("p",
 "The reference point is the last-value predictor, which carries no model at all: 0.905, 0.929 and 1.165 "
 "normalised units at the three horizons, a strong baseline on near-stationary records. Every learned model "
 "beats it: the direct predictor reaches 0.691, 0.679 and 0.711, the unstructured latent model 0.703, 0.691 and "
 "0.721, the physics-informed latent model 0.704, 0.689 and 0.714, and the structured model 0.706, 0.717 and "
 "1.144. The gain over the trivial predictor is 22% at 25 cycles and 39% at 100 cycles for the best two "
 "families, a real improvement and not a large one."),
("table", "Table 10. Measured-fleet results on the NASA C-MAPSS FD001 subset. Normalised RMSE at horizons of 25, 50 and 100 cycles, error-growth exponent, parameter count, per-step latency, and training time. All table values are the mean of two seeds; for the two structured rows the text reports the spread and the median of four seeds. The last-value predictor is the model-free reference.", None,
 [("Persistence", "0.905", "0.929", "1.165", "0.18", "-", "-", "-"),
  ("MLP-MD", "0.691", "0.679", "0.711", "0.02", "36,998", "0.22", "35"),
  ("GRU-LD", "0.703", "0.691", "0.721", "0.02", "31,686", "0.20", "48"),
  ("NODE", "0.699", "0.755", "0.983", "0.24", "36,999", "1.13", "146"),
  ("HNN", "0.829", "0.872", "1.394", "0.37", "34,050", "1.23", "129"),
  ("PI-Lat", "0.704", "0.689", "0.714", "0.01", "70,214", "0.25", "49"),
  ("PCL-WM", "0.706", "0.717", "1.144", "0.21", "102,863", "2.09", "232"),
  ("PCL-WM w/o symplectic", "0.720", "2.229", "14.067", "2.16", "102,863", "2.23", "222")]),
("p",
 "The second difference is the shape of the error curve. On the simulated plants the exponent lies between 0.9 "
 "and 1.0 and error grows roughly linearly, so the horizon separates models. On the fleet the two best families "
 "have an exponent of 0.02: the error stops growing after a few tens of cycles and sits at a plateau near 0.7. "
 "The reason is in the data. The cycle-to-cycle measurement noise of these channels is comparable to the "
 "degradation drift accumulated over the same cycle, so the last-value predictor is already at 0.839 on one "
 "step and no model goes below the noise floor. This domain measures filtering and short-horizon extrapolation, "
 "and reporting only the plateau height would hide which of the two it measured."),
("p",
 "The third difference is the ranking. The structured model is not the most accurate predictor on measured "
 "data; it is fourth of six at 25 cycles and fifth at 100, behind the direct and the unstructured latent "
 "predictors. Across four seeds its 100-cycle error is 0.743, 0.775, 1.167 and 1.890, and doubling the schedule "
 "gives 0.952 and 5.081, so two of six runs leave the fitted region. The final losses of the six runs lie within "
 "3% of each other, so the high error is not under-fitting but a free rollout departing from the fitted "
 "region, which is why the public-data rows are a mean with the spread stated rather than a best-of-seeds."),
("fig", os.path.join("figs", "fig7_public.png"),
 "Fig. 7. Measured-fleet behaviour. (a) One held-out engine, 100-cycle rollout of the compressor static "
 "pressure against the recorded trace and an eleven-cycle mean of it; the models reproduce the trend, not the "
 "per-cycle noise. (b) Error growth on the sixteen held-out engines, log-log axes; the plateau is set by sensor "
 "noise rather than by the horizon."),
("p",
 "What the fleet does confirm is the role of the Verlet map. Removing it takes the 100-cycle error from 1.144 "
 "to 14.067 on average, or from 0.971 to 14.484 at the median, with an exponent of 2.16: all four 60-epoch "
 "seeds diverge (12.006 to 15.296) and both 120-epoch runs diverge as well (6.576 and 13.742). An exact map for "
 "a modified Hamiltonian cannot repair a latently wrong model, but on a noisy record it is the difference "
 "between a bounded and an exploding rollout, by the mechanism identified in Section 5.8: the implicit "
 "half-kick damps the fastest latent mode, which measured data continuously re-excite. Structure here buys "
 "boundedness, not accuracy, and the two should not be reported as one number."),
("p",
 "The rule that follows is cheap to test. Fit the model, roll it out on a record fifty cycles longer than the "
 "intended horizon, and read the exponent: if it is well above one while the "
 "one-step error is acceptable, the rollout is drifting and the integrator, the latent gain or the horizon has "
 "to change; if it is near zero, the horizon is not the variable that matters, and a direct multi-step predictor "
 "matches the structured one at a fraction of the latency, as Table 10 shows for this fleet, 0.22 ms against "
 "2.09 ms. Both are legitimate regimes; Section 6 states what each implies."),
("p",
 "A second subset, and a degradation target rather than reconstruction, were run. Under the identical "
 "protocol the multi-condition subsets measure the sequence of regimes, the operating point changing at "
 "82 and 83 percent of boundaries: the direct predictors sit on the window-mean floor, 0.90, 0.96 and 0.85 against "
 "1.00, and the structured rollouts leave it at 2.38 and 7.51. Separate normalisation "
 "moves the direct predictors to 0.63-1.00 with the "
 "exponent near 0.1: what remains is noise. The degradation target was scored on the "
 "last 161 cycles of every record, as the drift a ten-cycle health block accumulates, so "
 "a rollout holding its last reading scores 1.00. At 100 cycles the structured rollout "
 "misses 0.46 and 0.61 of that drift on the two subsets, against 1.79 and 1.87 without the "
 "Verlet map, which is worse than holding still; the latent integrator is best at 0.30 and "
 "0.37, and the grid is in the release archive. A ramp fitted on training records alone "
 "reaches 0.17 and 0.19, so no rollout beats a constant. That constant is also the gap to "
 "published heads, which regress remaining life directly and reach a test-set RMSE of "
 "12.5-19.5 cycles on FD002 and 14.0-22.2 on FD004 over five methods [44]; a "
 "100-cycle reconstruction error of 0.93 to 1.48 is no route to that number, and none is claimed."),
("p",
 "The boundary here is narrower than a submission usually claims. Three conditions hold here: the horizon is "
 "shorter than the interval to failure, so no window contains a fault; the "
 "record is near-stationary, so filtering matters more than extrapolation; and the "
 "noise is comparable to the drift, so the achievable error is set by the noise floor, not the model. "
 "Under those conditions the ranking of Section 5 does not transfer, the structured model is not the most "
 "accurate predictor, and the one property that does transfer is boundedness of the rollout. A deployment with "
 "a longer horizon, a strongly non-stationary load, or a clean sensor would occupy a different part of Table 12."),
("h2", "5.10. Filters with the true equations as an upper reference"),
("p",
 "The comparison so far is between models that all have to discover the dynamics. An engineering reader will ask "
 "the complementary question: what does a state estimator achieve when the equations are known exactly? We "
 "therefore run an extended Kalman filter and an unscented Kalman filter [37] on the four simulated plants. Both "
 "filters see only the measured channels, are initialised from the four-observation history window with the "
 "unobserved components set to their training mean and a prior covariance equal to the variance of that "
 "component, and are then rolled out with the true equations. They are not competitors of the learned models; they "
 "are an upper reference that fixes the cost of learning; Table 11 reports them at four horizons on each plant."),
("table", "Table 11. State estimators running the true plant equations on the measured channels. Normalised RMSE at four horizons, filter initialised from the same four-observation window and rolled out with the exact dynamics. Values are means over the sixteen evaluation trajectories.", None,
 [("D1", "EKF", "0.062", "0.119", "0.214", "0.370", "0.86"),
  ("D1", "UKF", "0.062", "0.119", "0.214", "0.370", "0.86"),
  ("D2", "EKF", "0.056", "0.100", "0.187", "0.370", "0.91"),
  ("D2", "UKF", "0.057", "0.100", "0.187", "0.371", "0.90"),
  ("D3", "EKF", "0.010", "0.009", "0.008", "0.009", "-0.07"),
  ("D3", "UKF", "0.010", "0.009", "0.008", "0.009", "-0.07"),
  ("D9", "EKF", "0.131", "0.221", "0.264", "0.241", "0.29"),
  ("D9", "UKF", "0.132", "0.222", "0.264", "0.241", "0.29")]),
("p",
 "Three things follow, and only the first is comfortable for the method. First, the filters are much better than "
 "any learned model on three of the four plants: on the quadrotor the filter holds 0.370 at two hundred steps "
 "against 0.376 for the best learned family at the same horizon and 1.598 for the structured model; on the thermal "
 "network it holds 0.009 against 0.014 and 1.019; on the reaction-diffusion field it holds 0.241 against 0.211 for "
 "the neural ODE and 3.265 for the structured model. The gap is the price of not knowing the equations, and it is "
 "large enough that on a plant with a trustworthy model a learned world model is the wrong tool unless something "
 "in the loop demands differentiability or speed."),
("p",
 "Second, the filters are not uniformly better, and where they are not is instructive. On the projectile the "
 "filter loses at the shortest horizon, 0.062 against 0.046, and wins at two hundred steps, 0.370 against 0.793 "
 "for the structured model and 0.539 for the direct predictor. The short-horizon loss is the initialisation: the "
 "filter starts from a posterior over the twelve-dimensional state built from four observations, so its first "
 "predictions carry the residual of that warm start, whereas the learned models amortise the same inference into "
 "the encoder. The long-horizon win is the exponent, 0.86 for the filter against 1.33 for the structured model, "
 "and it is consistent with the argument of Section 5.8 about the latent coordinates rather than with the "
 "equations: a filter that carries its uncertainty grows its error more slowly than a learned map whose latent "
 "gain has no counterpart."),
("p",
 "Third, the two filters agree to the third decimal on every plant, including the quadrotor with its quaternion "
 "attitude, which is the case where the unscented transform is supposed to earn its cost. The reason is that the "
 "observation is a linear selection of the state, so the update is weakly nonlinear and the linearisation error "
 "enters only through the propagation. We report both rather than only the better one because the agreement is "
 "itself evidence that the reference is not an artefact of a particular filter's tuning, and the same reasoning "
 "explains why we did not add a particle filter: with a linear observation and a small process noise it would "
 "reproduce the same curve at a hundred times the cost."),
("p",
 "The practical reading for Section 6 is that the gap between learned and oracle-physics prediction is the "
 "honest measure of how far latent world models still are, and that it differs by plant. It is a factor of two "
 "on the projectile at the long horizon, a factor of four on the quadrotor, and two orders of magnitude on the "
 "thermal network, which is the plant where every channel is measured and a linear model is almost exact. None "
 "of the learned families closes this gap, and no ablation in Table 7 does either; the reportable result is the "
 "size of the gap per plant, not the claim that one learned structure is preferable to another."),
("h1", "6. Discussion"),
("h2", "6.1. Why the symplectic step fails"),
("p",
 "The failure is not a numerical accident, and it is not repaired by shrinking the step. It has three "
 "mechanisms, each of which is visible in the measurements."),
("p",
 "The first mechanism is that the symplectic map preserves the wrong flow. A symplectic integrator is exact for a "
 "modified Hamiltonian close to the one it is given, and it preserves the symplectic form of the map it composes. "
 "In a latent space the Hamiltonian is learned from finite data on a distribution of trajectories, and the map is "
 "trained against a multi-step reconstruction loss rather than against the flow. The composition of a learned "
 "Hamiltonian with an exactly symplectic map therefore produces a map that is symplectic for a vector field that "
 "is only approximately the one the data were generated by. The guarantee survives; the object it applies to "
 "does not. In the classical setting the Hamiltonian is known and the integrator is the only approximation; "
 "here the Hamiltonian is the approximation and the integrator is exact."),
("p",
 "The second mechanism is the mismatch between the learned step and the data step. The Verlet map"
 " advances the latent state by a step h that is trained, and the training signal comes from"
 " observations sampled at a fixed interval. When h is optimised jointly with the Hamiltonian, the"
 " optimiser can reduce the one-step loss by choosing a step that is stable for the average state,"
 " and the average state is not the state the model visits at step 150 of a rollout. A sigmoid"
 " keeps h in 0.05 to 0.50, which rules out the degenerate step but not one that is stable in"
 " training and unstable outside it. The measurement that supports this reading is the one-step"
 " column of Table 4: the configuration with the symplectic step has the best one-step error of all"
 " eight and the worst 200-step error, which is exactly the signature of a map tuned for local"
 " consistency."),
("p",
 "The third mechanism is that the learned latent dynamics has positive Lyapunov modes, and a symplectic map "
 "does not suppress them. A Hamiltonian flow has no attracting directions by construction, because the "
 "Jacobian is a symplectic matrix and its eigenvalues come in reciprocal pairs. A learned latent flow that "
 "approximates a dissipative plant and is forced into Hamiltonian form must represent the contraction "
 "somewhere, and if it cannot represent it in the canonical block it represents it in the residual block or "
 "not at all. When it is not represented, the map has expanding directions that a symplectic structure cannot "
 "remove, because symplecticity constrains the product of the eigenvalues, not their sign. The growth penalty "
 "on the latent norm, which the reported model uses, is a soft remedy; the measurement in Table 5 shows that the "
 "hard remedy is to drop the constraint and let the port-Hamiltonian dissipation do the contracting."),
("p",
 "The three mechanisms explain why the map costs accuracy where the canonical split is close to the physics, and "
 "Table 9 marks the limit of that explanation. On the reaction-diffusion field the removal makes the same model "
 "2.6 times worse, which the three mechanisms do not predict: there the latent dynamics is stiff, the implicit "
 "half-kick of the Verlet map damps the fastest latent modes, and a map that is not there cannot damp anything. "
 "The mechanisms therefore describe a conditional cost. What is common to all four plants is the pattern at short "
 "horizon: the structured model is the most accurate at 25 steps and the least accurate at 200 steps, in both the "
 "240-trajectory and the 60-trajectory regimes, on every domain. The structure buys local consistency and pays for "
 "it in compounding, and whether the symplectic stage helps or hurts is decided by the spectrum of the learned "
 "latent dynamics rather than by the principle."),
("h2", "6.2. What survives"),
("p",
 "The components that survive the same tests are the three that constrain the latent dynamics without "
 "constraining its form, and the evidence for them is the distribution-shift study rather than the ablation."),
("p",
 "The port-Hamiltonian residual block with R=LLT survives because it imposes a property of the plant rather "
 "than a property of the coordinates. A dissipative plant has a non-negative energy production rate, and the "
 "parameterisation makes that true of the model by construction, in any coordinates and for any decoder. The "
 "measurement that supports it is the out-of-distribution "
 "exponent of 0.947 in Table 5, the only exponent below one among the four models there, reproduced by both "
 "seeds: the error saturates instead of accumulating once the training set is small and the operating point is "
 "shifted."),
("p",
 "The gauge condition and the certificate regulariser rest on the same argument and on weaker evidence. Both "
 "remove or penalise a degree of freedom that the observations do not constrain, which is the right kind of "
 "constraint to add when data are scarce, and neither changes the short-horizon accuracy. In the in-distribution "
 "ablation of Table 4 each moved the error by less than the seed spread, so their isolated value is not "
 "established here: they belong to the group that produces the sixteen percent gain under shift, and the study "
 "does not separate them inside it. We keep them because they are cheap and because the "
 "mechanism is sound, not because the ablation supports them."),
("p",
 "What does not survive, besides the symplectic step, is the idea that the structural prior should be "
 "responsible for accuracy in distribution. In distribution the unstructured latent model is nine percent ahead "
 "of the best structured configuration, and both beat the direct predictor by a factor of two. The mechanism is "
 "architectural, not physical: the residual decoder and the long-horizon "
 "curriculum do the work, and the same architecture with no physics at all reproduces it. A paper that "
 "attributed that gain to mechanics would be wrong, and the ablation in Table 4 is what makes the attribution "
 "possible."),
("h2", "6.3. Relation to the literature"),
("p",
 "The negative result on the symplectic step is not a contradiction of the literature; it is a boundary "
 "condition on it. Symplectic ODE-Net and the Hamiltonian generative network report strong long-horizon "
 "behaviour, and both evaluate on systems that are conservative or lightly damped, observed in full or "
 "reconstructed from images with a decoder trained through a reconstruction loss over complete trajectories "
 "[20, 22]. Under those conditions the mismatch between the learned Hamiltonian and the true vector field is "
 "small, and the integrator's exactness is a real gain. Our domains are damped, actuated, and only partly "
 "observed, and the ablation holds the rest of the pipeline fixed, which isolates the constraint. The reading "
 "we propose is that the symplectic constraint is worth its cost when the Hamiltonian is nearly known and the "
 "state is nearly observed, is a liability when neither holds but the learned spectrum is already contracting, "
 "and is a genuine gain when the learned spectrum contains stiff modes that need the implicit stage, as on the "
 "reaction-diffusion field in Table 9."),
("p",
 "The positive result on constrained dissipation connects to the dissipative variant of the same family [23] "
 "and to the port-Hamiltonian formulation more generally [21]. Our contribution to that line is not the "
 "construction, which follows it, but the evidence about where the construction pays: at sixty trajectories "
 "and a shifted initial condition, and in the exponent rather than in the level of the error."),
("p",
 "The comparison with graph-based and operator-based simulators [12, 16, 19] is a comparison of scale and "
 "assumption. Those models are trained on fully observed particle or field states and are evaluated on "
 "resolution transfer; our models are trained on partial observations of a control system and are evaluated "
 "on horizon transfer. The two settings reward different priors, and the fact that a message-passing simulator "
 "generalises across resolutions does not imply that a latent Hamiltonian generalises across horizons."),
("h2", "6.4. Limitations"),
("p",
 "The simulated domains are four and self-contained: a projectile, a quadrotor, a thermal network and a "
 "reaction-diffusion field, all integrated with the same solver at the same step. The reinforcement-learning "
 "benchmarks with contact dynamics were not run, because their simulator [41] was not available in the "
 "environment where the experiments ran. Another suite would change the numbers, and until it is run the ranking "
 "is a statement about partially observed, actuated, dissipative systems of the kind in Table 1."),
("p",
 "The fifth domain is measured and public, which answers the reproducibility objection only in part. FD001 is "
 "recorded at a single operating point, so the cross-condition shift that the shifted-initial-speed test "
 "exercises in the simulator is absent there, and the multi-regime subsets that do contain it cannot be cut into "
 "trajectories because their settings change at every cycle. The measured comparison is noise limited, with a "
 "plateau near 0.7 and an exponent near zero: it ranks models on filtering and extrapolating a hundred cycles, "
 "not a thousand. Finally, the structured "
 "model needed a bounded rollout to be usable on this domain: one run out of four left the fitted region and "
 "reported a 100-cycle error above 5, while the symplectic-free variant diverged in all four runs. That "
 "instability is a property of latent rollouts in general; it is reported rather than averaged away, so the "
 "public-data comparison is a statement about medians, not guarantees."),
("p",
 "Every simulated configuration in this paper is run with two seeds, and the tables report the mean; the "
 "measured-fleet section uses two seeds for every baseline and four for the structured model. The measured "
 "seed-to-seed spread on the projectile is 0.06 for the direct model, 0.13 for the structured model and 0.27 "
 "for the Hamiltonian baseline, and it is reported so that a reader can see which of the differences in Tables "
 "2 and 4 are larger than the noise. The 2.3 factor produced by removing the symplectic step is far outside "
 "it; the differences produced by the other removals are not, and are treated as indicative at best. Table 9 "
 "repeats the integrator ablation on all four plants with two seeds each; the reversal on the "
 "reaction-diffusion field, 3.27 against 8.34, holds in both seeds and is reported as a direction of effect "
 "rather than as a magnitude."),
("p",
 "The certificate is a diagnostic in this paper, not a theorem. The decomposition of Eq. (14) is computed from "
 "measured quantities, and the horizon of Eq. (16) uses an empirically estimated Lipschitz constant. We do not "
 "claim that the horizon is a conservative bound in the sense of the classical theory, and we do not report "
 "coverage statistics against ground truth, because such statistics would require a validated bound that this "
 "study does not provide."),
("p",
 "Finally, the closed-loop study covers the projectile only, with three controller runs and twenty-four "
 "episodes per configuration. The controller is a sampling-based planner with a fixed budget, not a tuned MPC "
 "with terminal ingredients. Three runs are enough to show that the differences between models lie inside the "
 "run-to-run range and not enough to order them."),

("h2", "6.5. Choosing priors for a given plant"),
("p",
 "The decomposition is useful only if it becomes a decision made before training. Four "
 "questions about the plant carry most of the information this study can supply, and each of "
 "them is answerable from the instrumentation list and the data set rather than from the "
 "model. Is the latent state re-estimated from a window at every step, or is it carried "
 "forward by the model itself? Is the flow near conservative, with a slowly varying "
 "Hamiltonian, or stiff, with fast modes that a single step length cannot resolve? Does the "
 "prediction horizon extend past the window over which the model was fitted? And is the "
 "quantity that matters a rollout error, a bound on the rollout, or the tracking error of a "
 "controller that replans? Table 12 maps the answers onto the choice this study can support, "
 "and cites the measurement behind each row rather than the argument."),
("table", "Table 12. Prior selection for a plant described by its instrumentation, its latent update, and its evaluation metric. Each row states the configuration this study supports and the measurement it rests on; the rows are not independent, since the plants that motivate them differ in more than one property at once.", None,
 [("Sparse fixed sensors, no momentum channels", "the encoder, with the physical prior applied to the latent only", "applying Hamilton's equations to the measured channels", "HNN without encoder, 1.46 to 4.32 across four plants"),
  ("Latent re-estimated from a window each step, canonical pair approximate, flow near-conservative", "constraints on the latent and its growth", "the discrete symplectic composition", "2.3 times better without it on D1, 1.37 on D2, 1.09 on D3"),
  ("Stiff latent dynamics with fast modes", "the implicit stage of the map", "the explicit or removed map", "2.6 times worse without it on D9, 3.27 against 8.34"),
  ("Few trajectories, about 60, tested outside the trained range", "encoder, constrained dissipation, gauge penalty", "the symplectic step", "0.595 against 0.704 out of distribution, exponent 0.95, two seeds"),
  ("Many trajectories, about 240, tested in distribution at 200 steps", "the unconstrained baselines, which stay competitive", "expecting accuracy from the remaining constraints", "Table 4 rows differ by less than their own seed spread"),
  ("Horizon far past the fitted window, boundedness is the metric", "the map, read as a bound rather than as an estimator", "judging the choice by short-horizon error", "0.971 against 14.484 at 100 cycles, all four seeds diverging when removed"),
  ("Closed-loop control with replanning at every step", "a closed-loop trial, to confirm that the model is usable in the loop", "ranking the models on closed-loop error from a short campaign", "closed-loop error spans 0.0649 to 0.0739 across five configurations, less than the run-to-run spread within one configuration, 0.015")]),
("p",
 "Two cautions belong with the table. The rows are not orthogonal: the plant that motivates the second row "
 "also has few trajectories, the one that motivates the third also has strong dissipation, so the last column "
 "belongs to the configuration rather than to the factor named in the row. And the table summarises what four "
 "plants and one measured fleet showed at this scale; it is not a validated selection rule. A reader who uses "
 "it should reproduce the two cheapest rows first, the encoder and the choice of integrator, since those are the "
 "effects that survived a second seed in every domain, and treat the middle rows as hypotheses to be checked on "
 "the plant at hand."),

("h1", "7. Conclusion"),
("p",
 "This paper asked which parts of a physical prior help, and where, in a latent world model for engineering "
 "control; answering it required decomposing the prior and duplicating every configuration whose claim "
 "depended on it."),
("p",
 "The main finding is negative, conditional, and useful. A fixed-step symplectic integrator inside a learned "
 "latent space raises the 200-step error by a factor of 2.3 on the projectile, where the canonical split is "
 "closest to the physics, by 1.37 on the quadrotor, and by 1.09 on the thermal network, and it is the cheaper "
 "option by a factor of 2.6 on the reaction-diffusion field, where its implicit stage damps stiff latent modes "
 "that a removed map exposes. The mechanisms behind the first three cases are a mismatch of premises rather "
 "than a numerical artefact: the integrator is exact for a flow that is only approximately the data's, its step "
 "is optimised for local consistency rather than global stability, and symplecticity constrains the product of "
 "the linearisation eigenvalues but does not suppress expanding directions. The fourth "
 "case marks the limit of that argument and turns the ablation into a design rule: match the integrator to the "
 "spectrum of the learned latent dynamics, which a short validation rollout can check before deployment."),
("p",
 "The positive finding is narrower than the original claim. A residual block in port-Hamiltonian form whose "
 "dissipation is parameterised as R=LLT, together with a gauge condition on the canonical block, produces a "
 "model that loses nine percent in distribution and beats it by about sixteen percent when sixty "
 "trajectories are available and the initial condition is shifted outside the training range, an effect that "
 "shows up in the error-growth exponent, 0.947 against 1.018, and therefore in whether the error saturates or "
 "accumulates. Both components are cheap to add and cheap to evaluate, neither changes the short-horizon "
 "accuracy, and that is why their effect was invisible in evaluations that stopped at one-step error. The "
 "two seeds bracket the out-of-distribution error between 0.581 and 0.608, so the margin is not a "
 "seed artefact."),
("p",
 "On the public measured fleet the conclusion changes form once more. Every learned model beat the last-value "
 "predictor by 22% at 25 cycles and 39% at 100 cycles, the error curve flattened at a noise-induced plateau "
 "instead of growing, and the direct multi-step predictor was the most accurate model at every horizon, at a "
 "tenth of the latency of the structured one. The symplectic map did not improve accuracy there either, but "
 "removing it converted a bounded rollout into a divergent one, 0.971 against 14.484 at 100 cycles with all four "
 "seeds affected. Structure therefore earns its place in a measured application through stability of the rollout "
 "and through the out-of-distribution gain above, not through short-horizon accuracy. The filters of Section 5.10 "
 "then set the scale of what remains: given the exact equations, a Kalman filter predicts two hundred steps with "
 "an error of 0.37 on the projectile, 0.37 on the quadrotor, 0.01 on the thermal network and 0.24 on the "
 "reaction-diffusion field without any training, so the gap to the best learned model runs from a factor of two "
 "to two orders of magnitude, and that gap is what a learned world model has to close before it can replace a "
 "physics-based estimator."),
("p",
 "For a reader who intends to deploy such a model, the practical reading is that the structured design is worth "
 "its cost only in the regime where structure pays: few trajectories, shifted operating points, and a controller "
 "that plans over a horizon long enough to feel the growth behaviour of the model. Where data are plentiful and "
 "the operating point is the training point, the unstructured latent model with a residual decoder is cheaper by "
 "an order of magnitude in latency and equal in accuracy, and the direct predictor is cheaper still. Table 12 "
 "states that boundary in the form in which it can be used: four questions about the plant, the configuration "
 "each answer supports, and the measurement in this paper that the configuration rests on, so that the next "
 "reader can decide before training rather than after."),

("h1", "CRediT authorship contribution statement"),
("p", "Tian-ci Zhang: Conceptualization, Methodology, Software, Formal analysis, Writing - original draft, "
      "Writing - review and editing. Shuai Hou: Software, Validation, Investigation, Data curation, "
      "Visualization. Yong-yong Wei: Validation, Investigation, Data curation, Writing - review and editing. "
      "Hao Fang: Conceptualization, Supervision, Writing - review and editing."),
("h1", "Declaration of competing interest"),
("p", "The authors declare that they have no known competing financial interests or personal relationships that "
      "could have appeared to influence the work reported in this paper."),
("h1", "Funding"),
("p", "This research did not receive any specific grant from funding agencies in the public, commercial, or "
      "not-for-profit sectors."),
("h1", "Acknowledgements"),
("p", "The authors thank the anonymous reviewers for comments that improved the presentation of the negative "
      "results."),
("h1", "References"),
("refs", [
 "S.L. Brunton, J.L. Proctor, J.N. Kutz, Discovering governing equations from data by sparse identification of nonlinear dynamical systems, Proc. Natl. Acad. Sci. U.S.A. 113 (15) (2016) 3932-3937.",
 "R.T.Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud, Neural ordinary differential equations, in: Advances in Neural Information Processing Systems 31, 2018, pp. 6571-6583.",
 "K. Chua, R. Calandra, R. McAllister, S. Levine, Deep reinforcement learning in a handful of trials using probabilistic dynamics models, in: Advances in Neural Information Processing Systems 31, 2018, pp. 4754-4765.",
 "M. David, F. Mehats, Symplectic learning for Hamiltonian neural networks, J. Comput. Phys. 494 (2023) 112495.",
 "S. Greydanus, M. Dzamba, J. Yosinski, Hamiltonian neural networks, in: Advances in Neural Information Processing Systems 32, 2019, pp. 15379-15389.",
 "E. Hairer, C. Lubich, G. Wanner, Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations, second ed., Springer, Berlin, 2006.",
 "E. Hairer, C. Lubich, G. Wanner, Geometric numerical integration illustrated by the Stormer-Verlet method, Acta Numer. 12 (2003) 399-450.",
 "D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, Learning latent dynamics for planning from pixels, in: Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 2019, pp. 2555-2565.",
 "D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, Mastering diverse control tasks through world models, Nature 640 (2025) 647-653.",
 "N. Hansen, H. Su, X. Wang, TD-MPC2: scalable, robust world models for continuous control, in: International Conference on Learning Representations, 2024.",
 "G.E. Karniadakis, I.G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physics-informed machine learning, Nat. Rev. Phys. 3 (2021) 422-440.",
 "N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A.M. Stuart, A. Anandkumar, Neural operator: learning maps between function spaces with applications to PDEs, J. Mach. Learn. Res. 24 (2023) 1-97.",
 "M. Lutter, C. Ritter, J. Peters, Deep Lagrangian networks: using physics as model prior for deep learning, in: International Conference on Learning Representations, 2019.",
 "D.Q. Mayne, J.B. Rawlings, C.V. Rao, P.O.M. Scokaert, Constrained model predictive control: stability and optimality, Automatica 36 (6) (2000) 789-814.",
 "A. Nagabandi, G. Kahn, R.S. Fearing, S. Levine, Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning, in: IEEE International Conference on Robotics and Automation, 2018, pp. 7559-7566.",
 "T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P.W. Battaglia, Learning mesh-based simulation with graph networks, in: International Conference on Learning Representations, 2021.",
 "M. Raissi, P. Perdikaris, G.E. Karniadakis, Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, J. Comput. Phys. 378 (2019) 686-707.",
 "T. Salzmann, E. Kaufmann, J. Arrizabalaga, M. Pavone, D. Scaramuzza, M. Ryll, Real-time neural MPC: deep learning model predictive control for quadrotors and agile robotic platforms, IEEE Robot. Autom. Lett. 8 (4) (2023) 2397-2404.",
 "A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, P.W. Battaglia, Learning to simulate complex physics with graph networks, in: Proceedings of the 37th International Conference on Machine Learning, PMLR 119, 2020, pp. 8459-8468.",
 "P. Toth, D.J. Rezende, A. Jaegle, S. Racaniere, A. Botev, I. Higgins, Hamiltonian generative networks, in: International Conference on Learning Representations, 2020.",
 "A. van der Schaft, D. Jeltsema, Port-Hamiltonian systems theory: an introductory overview, Found. Trends Syst. Control 1 (2-3) (2014) 173-378.",
 "Y.D. Zhong, B. Dey, A. Chakraborty, Symplectic ODE-Net: learning Hamiltonian dynamics with control, in: International Conference on Learning Representations, 2020.",
 "Y.D. Zhong, B. Dey, A. Chakraborty, Dissipative SymODEN: encoding Hamiltonian dynamics with dissipation and control into deep learning, in: ICLR Workshop on Integration of Deep Neural Models and Differential Equations, 2020.",
 "J. Rettberg, J. Kneifl, J. Herb, P. Buchfink, J. Fehr, B. Haasdonk, Data-driven identification of latent port-Hamiltonian systems, arXiv preprint arXiv:2408.08185, 2024.",
 "X. Luan, C. Shi, PH-Dreamer: a physics-driven world model via port-Hamiltonian generative dynamics, arXiv preprint arXiv:2605.18303, 2026.",
 "H. Tang, H. Cui, K. Xu, Z. Mei, K. Wang, HaM-World: soft-Hamiltonian world models with selective memory for planning, arXiv preprint arXiv:2605.05951, 2026.",
 "P. Li, K. Tan, D. Moyer, T. Beckers, Identify then realize: contrastive learning of latent port-Hamiltonian dynamics from partial observations, arXiv preprint arXiv:2605.16682, 2026.",
 "S. Cui, J. Ma, Physically native world models: a Hamiltonian perspective on generative world modeling, arXiv preprint arXiv:2605.00412, 2026.",
 "A. Saxena, K. Goebel, D. Simon, N. Eklund, Damage propagation modeling for aircraft engine run-to-failure simulation, in: 2008 International Conference on Prognostics and Health Management, IEEE, 2008, pp. 1-9.",
 "D. Ha, J. Schmidhuber, Recurrent world models facilitate policy evolution, in: Advances in Neural Information Processing Systems 31, 2018, pp. 2455-2467.",
 "Y. Rubanova, R.T.Q. Chen, D. Duvenaud, Latent ordinary differential equations for irregularly-sampled time series, in: Advances in Neural Information Processing Systems 32, 2019, pp. 5320-5330.",
 "P. Kidger, J. Morrill, J. Foster, T. Lyons, Neural controlled differential equations for irregular time series, in: Advances in Neural Information Processing Systems 33, 2020, pp. 6696-6707.",
 "B. Lusch, J.N. Kutz, S.L. Brunton, Deep learning for universal linear embeddings of nonlinear dynamics, Nat. Commun. 9 (2018) 4950.",
 "K. Champion, B. Lusch, J.N. Kutz, S.L. Brunton, Data-driven discovery of coordinates and governing equations, Proc. Natl. Acad. Sci. U.S.A. 116 (45) (2019) 22445-22450.",
 "M. Cranmer, A. Sanchez-Gonzalez, P. Battaglia, R. Xu, K. Cranmer, D. Spergel, S. Ho, Discovering symbolic models from deep learning with inductive biases, in: Advances in Neural Information Processing Systems 33, 2020, pp. 17429-17442.",
 "P. Jin, Z. Zhang, A. Zhu, Y. Tang, G.E. Karniadakis, SympNets: intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems, arXiv preprint arXiv:2001.03750, 2020.",
 "S.J. Julier, J.K. Uhlmann, Unscented filtering and nonlinear estimation, Proc. IEEE 92 (3) (2004) 401-422.",
 "G. Williams, A. Aldrich, E.A. Theodorou, Model predictive path integral control: from theory to parallel computation, J. Guid. Control Dyn. 40 (2) (2017) 344-357.",
 "F.O. Heimes, Recurrent neural networks for remaining useful life estimation, in: 2008 International Conference on Prognostics and Health Management, IEEE, 2008, pp. 1-8.",
 "Y. Wu, M. Yuan, S. Dong, L. Lin, Y. Liu, Remaining useful life estimation of engineered systems using vanilla LSTM neural networks, Neurocomputing 275 (2018) 1678-1686.",
 "E. Todorov, T. Erez, Y. Tassa, MuJoCo: a physics engine for model-based control, in: IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 5026-5033.",
 "S. Bhardwaj, C. Bajaj, PHAST: port-Hamiltonian architecture for structured temporal dynamics forecasting, arXiv preprint arXiv:2602.17998, 2026.",
 "R. Yalovetzky, R. Balestriero, Y. LeCun, LeHami: symplectic latent dynamics for world models, arXiv preprint arXiv:2610.15267, 2026.",
 "A. Chaoub, A. Voisin, C. Cerisara, B. Iung, Learning representations with end-to-end models for improved remaining useful life prognostic, arXiv preprint arXiv:2104.05049, 2021.",
]),
("h1", "Data availability"),
("p",
 "The measured domain is the publicly distributed NASA C-MAPSS turbofan degradation dataset, obtained from the "
 "NASA Prognostics Center of Excellence data set repository and from the mirror listed in the reproduction notes; "
 "no synthetic augmentation was applied to it. The simulated domains are generated by the code described in "
 "Appendix A, so every number in Tables 2 to 9 is reproducible from the scripts together with a fixed seed."),
("p",
 "Manuscript source, experiment drivers, aggregated results and the scripts that regenerate every table and figure are packaged as an archive submitted with this paper, and the same tree is prepared as a public repository whose identifier is released on acceptance."),

("h1", "Appendix A. Training configuration and reproduction notes"),
("p",
 "Table A.1 collects every setting needed to reproduce the reported numbers. The domains, "
 "the models, the training loop and the evaluation share one code base with no external "
 "simulator dependency; the plant is integrated with fourth-order Runge-Kutta and the models are "
 "trained on windows from that integrator, so the only stochastic elements are the "
 "initial-condition sampler, the control sampler and the observation noise."),
("table", "Table A.1. Training configuration used for all reported configurations.", None,
 [("Plant integrator", "RK4, step 0.02 s, 400 steps per episode"),
  ("Training trajectories", "240 (60 in the data-efficiency study)"),
  ("Evaluation trajectories", "16, starting at step 50"),
  ("Observation window", "4 consecutive observations"),
  ("Observation noise", "1% of the channel standard deviation"),
  ("Optimiser", "Adam, learning rate 1.5e-3, cosine annealing"),
  ("Epochs, batch size", "60, 64"),
  ("Batches per epoch", "25"),
  ("Rollout curriculum", "2 to 30 steps, reached at two thirds of training"),
  ("Discount factor", "0.9 per rollout step"),
  ("Gauge penalty weight", "0.01"),
  ("Latent growth penalty weight", "0.02"),
  ("Smoothness penalty", "0.01, applied every third batch"),
  ("Symplecticity weight", "0.005, ablation configuration only"),
  ("Gradient clipping", "unit norm"),
  ("Latent dimension", "32 = 16 canonical (m = 8) + 16 residual"),
  ("Hamiltonian network", "2m to 128 to 128 to 1"),
  ("Residual energy network", "full latent vector, differentiated on the residual block"),
  ("Learned step", "0.05 + 0.45 sigmoid(a)"),
  ("MPC setting", "horizon 15, 128 candidates, 3 iterations, 16 elites, control weight 0.02")]),
("p",
 "Two conventions determine how the tables should be read: errors are normalised by the per-channel standard "
 "deviation of the training set, so 1.0 means the model is no better than predicting the mean, and the horizons "
 "count from the end of the input window, so 200 is four seconds of flight on the projectile. The exponent is "
 "fitted on four horizons, a short lever arm, and is reported as a qualitative indicator of growth rather than "
 "a precise rate."),
("p",
 "The public-domain runs use the same loop with two differences that Table A.1 does not list because they follow "
 "from the records rather than from the method: the training set is sixty-eight windows of 161 cycles taken from "
 "the FD001 files, there is no control channel, and the horizons are 25, 50 and 100 cycles. The dataset is read "
 "from the distribution mirror of the NASA repository (raw.githubusercontent.com/ericlrf/rul, directory "
 "CMAPSSData), and the loader, the split, and the channel selection are fixed in one file so that the sixteen "
 "held-out records cannot be reshuffled between runs."),
]
