# -*- coding: utf-8 -*-
import os
"""Method and experimental protocol (Sections 3 and 4)."""

BLOCKS = [
("h1", "3. Method"),
("h2", "3.1. Problem setting and notation"),
("p",
 "We consider a plant with state $x\\in\\mathbb{R}^{n}$, control $u\\in\\mathbb{R}^{p}$, and a measured output "
 "vector $o$ whose dimension $n_o$ is smaller than $n$. Only a fixed subset of state channels is instrumented, so the output is "
 "a linear projection of the state plus noise, and the unmeasured channels include exactly the ones a "
 "first-principles model needs at inference time, namely velocities and body rates. The learning problem is to "
 "find a map that predicts the measured outputs $L+1$ steps ahead from a window of $L$ past observations "
 "and a control sequence, without ever being given the full state."),
("p",
 "Formally, given windows $o_{t-L:t}$ and controls $u_{t:t+K-1}$, the model produces "
 "$\\hat{o}_{t+1:t+K}$, and the training loss is the multi-step reconstruction error over the window. The "
 "quantities that matter downstream are the normalised root-mean-square error at a set of horizons, the "
 "exponent of the error-growth law, the robustness of both to a shift in initial conditions, and the tracking "
 "error when the model is placed inside a model predictive controller."),
("h2", "3.2. Latent decomposition"),
("p",
 "We refer to the model of Section 3.1 as PCL-WM. It encodes the observation window into a latent state "
 "partitioned into three blocks, a canonical "
 "pair and a residual vector, as in Eq. (1). The canonical pair carries the oscillatory part of the dynamics, "
 "which is the part a Hamiltonian description is meant to capture, and the residual vector carries everything "
 "else: actuation, non-conservative effects, and the part of the dynamics that is not expressible as a gradient "
 "of a scalar function of two variables. The split is a modelling choice, not an assumption about the plant; a "
 "system with no oscillatory component simply learns a flat Hamiltonian and lets the residual block do the work. "
 "The port-Hamiltonian reading of the residual block follows the standard account of dissipation and "
 "interconnection in physical systems [21]."),
("eq", 1),
("p",
 "The canonical block evolves under a learned Hamiltonian through Hamilton's equations, Eq. (2), and the "
 "residual block evolves under a port-Hamiltonian map with a learned interconnection and a learned dissipation, "
 "Eq. (5), driven by the control through a learned input matrix. Both blocks are integrated with the same "
 "learned step $h$, and both are recombined before decoding. The decoder is residual in the observation, "
 "Eq. (9), so that the identity map is available to the model and the network only has to represent the "
 "increment; this detail is not cosmetic, and Section 5.3 shows that it accounts for most of the improvement "
 "over an earlier non-residual version of the same model."),
("eq", 2),
("p",
 "For the canonical block we use the symmetric Störmer-Verlet composition of Eqs. (3) and (4), written as a "
 "half-kick on the momentum, a full drift on the position, and a second half-kick using the updated position. "
 "The map is second order, symplectic, and time reversible for the Hamiltonian it is applied to, and it is the "
 "map we later remove in the main ablation; Section 5.8 repeats the removal on all four domains, which is what "
 "separates the cost of the integrator from the cost of the canonical split."),
("eq", 3),
("eq", 4),
("p",
 "The residual block is written in port-Hamiltonian form with a skew-symmetric interconnection $J_r$ and a "
 "dissipation matrix $R$. The dissipation is parameterised as $R=LL^{\\top}$ with $L$ lower triangular, which "
 "makes $R$ symmetric positive semidefinite by construction rather than by penalty, Eq. (6). The "
 "interconnection is parameterised by its strictly lower triangle and antisymmetrised, so that the energy "
 "balance of Eq. (15) holds exactly for the learned parameters: the first term on the right-hand side is "
 "non-positive for every state, and the second is the supply rate through the input port."),
("eq", 5),
("eq", 6),
("p",
 "The canonical matrix $J$ of Eq. (7) is the block matrix that maps the gradient of the Hamiltonian to the "
 "velocity, with $e_i$ the canonical basis vectors. The complete one-step map is the composition of the two "
 "block updates and the input injection, Eq. (8), and it is this map, not the continuous flow, that is trained "
 "and rolled out."),
("eq", 7),
("eq", 8),
("p",
 "The decoder is residual in the previous observation, Eq. (9), where $D_\\psi$ is a two-layer network. "
 "Residual decoding removes the systematic bias of a decoder that has to reproduce the observation from scratch, "
 "and it also makes the one-step loss well conditioned at the start of training, because the identity "
 "prediction is the initial condition of the decoder rather than an accidental property of the weights."),
("eq", 9),
("fig", os.path.join("figs", "fig1_architecture.png"),
 "Fig. 1. Structure of PCL-WM and the evaluation it is put through. (a) A window of observations is "
 "encoded into a canonical pair and a residual block. The canonical pair is advanced by a learned Hamiltonian "
 "with a Verlet map; the residual block follows port-Hamiltonian dynamics with a dissipation matrix fixed to "
 "be positive semidefinite by the factorisation R=LLT. The two blocks are recombined and decoded residually, "
 "and the gauge condition removes the translation symmetry of the canonical coordinates. (b) The six "
 "questions the paper asks, each with the table that answers it."),
("h2", "3.3. Gauge fixing"),
("p",
 "The canonical block is defined only up to a translation: adding a constant vector to $q$ and the corresponding "
 "term to the potential leaves the equations of motion unchanged, and the learned Hamiltonian is free to drift "
 "along that flat direction. In a latent space the drift is not harmless, because the decoder is not invariant "
 "and the translation shows up as a slowly growing offset in the reconstructed observation. We therefore "
 "penalise the batch mean of the canonical position, Eq. (10), which is the discrete analogue of choosing the "
 "centre-of-mass frame. The penalty is cheap, it does not constrain the relative coordinates that carry the "
 "dynamics, and Section 5.3 measures its contribution."),
("eq", 10),
("p",
 "For diagnostics we also evaluate the symplecticity residual of the learned step map, Eq. (11), by applying "
 "the Jacobian of the map to random probes and forming the defect of the symplectic form. The residual is not "
 "used as a training loss in the reported runs; it is reported because it separates the claim that the map is "
 "symplectic from the claim that the map is accurate, and Section 6 uses the distinction."),
("eq", 11),
("h2", "3.4. Training objective"),
("p",
 "The objective is a discounted multi-step reconstruction loss over a rollout of length $K$, Eq. (12), with "
 "$\\gamma=0.9$. The rollout length is not fixed during training: it follows a curriculum that starts at two "
 "steps and reaches thirty, so that the model first learns a locally accurate map and only later is asked to "
 "stay accurate over the horizon it will be evaluated on. Without the curriculum the model optimises a one-step "
 "objective that is blind to the unstable modes of the learned latent dynamics, and Section 5.3 quantifies the "
 "consequence."),
("eq", 12),
("p",
 "Three regularisers are added. The gauge penalty of Eq. (10) is applied with weight 0.01. A growth penalty on "
 "the latent norm, the squared hinge of the excess of that norm over its initial value, averaged over the "
 "batch, is applied with weight 0.02; it suppresses the positive Lyapunov modes that a learned latent dynamics tends to "
 "acquire, and it is the difference between a rollout that stays bounded and one that diverges after roughly "
 "one hundred steps. A smoothness penalty on the step map, applied every third batch with weight 0.01, keeps the "
 "learned step close to its neighbours in the state space. Gradients are clipped at unit norm."),
("p",
 "The symplecticity residual of Eq. (11) is not part of the loss in the reported configuration. An earlier "
 "version of the model used it with weight 0.005, and the ablation reported below as the symplectic-free "
 "configuration removes it; the measured effect is small and is discussed with the other ablations."),
("h2", "3.5. Error decomposition and online certificate"),
("p",
 "The certificate rests on a one-step amplification argument. If the learned map is Lipschitz in the latent "
 "state with constant $L$ and the per-step error injected by the model is bounded by $\\delta$, then the error "
 "propagates linearly with the amplification factor $\\rho=1+Lh$, as in Eq. (13). The relation is not a claim "
 "about the plant; it is a budget that says how fast a per-step model error is allowed to grow before the "
 "prediction leaves the tolerance band."),
("eq", 13),
("p",
 "A rollout error can also be attributed, and the attribution is the practical value of the construction. "
 "Starting from the observation at the end of the window, the error after $K$ steps decomposes into five terms: "
 "the encoder error in recovering the latent state from a finite window, the decoder error in returning to "
 "observation space, the error of the learned Hamiltonian and residual energy, the error of the integrator "
 "relative to the exact flow, and the error of the data itself, which enters through noise and through the "
 "mismatch between the sampled step and the step the model is asked to take. The decomposition is written in "
 "Eq. (14) and is computed online by replaying the window, the latent trajectory, and the reconstruction, so "
 "that each term is a measured quantity rather than a symbolic placeholder."),
("eq", 14),
("p",
 "The decomposition is consistent with the energy structure of the residual block. Differentiating the learned "
 "energy along the port-Hamiltonian flow gives Eq. (15): the first term is non-positive for every state because "
 "$R$ is positive semidefinite, and the second is the power delivered through the input port. The identity holds "
 "exactly for the learned parameters, which is the sense in which the dissipation is constrained rather than "
 "penalised."),
("eq", 15),
("p",
 "The horizon at which the amplified error first exceeds a tolerance $\\varepsilon$ follows from the geometric "
 "series of Eq. (13), giving Eq. (16). The bound is conservative whenever $L$ is estimated over a region larger "
 "than the states the rollout visits, which is the usual case, and Section 5.2 reports the ratio of the "
 "certified horizon to the horizon at which the error crosses the tolerance."),
("eq", 16),
("p",
 "Two caveats are stated here rather than in the discussion. The Lipschitz constant is estimated empirically "
 "from finite differences along the rollout, so the bound is a posteriori in nature; and the certificate is "
 "computed for the horizon that the model was trained on, beyond which the extrapolation is not covered. We "
 "therefore use the certificate as a monitoring signal inside the controller, not as a guarantee, and the "
 "numbers in Section 5.6 are measurements of the decomposition, not of a proven bound."),
("h2", "3.6. Implementation"),
("p",
 "The latent dimension is 32, of which $2m=16$ are canonical ($m=8$) and $r=16$ are residual. The Hamiltonian "
 "is a three-layer network with 128 hidden units per layer, the residual energy is differentiated only on the "
 "residual coordinates, and the input map is one linear layer. The step $h$ is parameterised as $0.05+0.45\\,\\sigma(a)$ with a "
 "learned scalar $a$, which keeps it inside the interval where the Verlet map is stable for the learned "
 "Hamiltonian and prevents the optimiser from buying accuracy by shrinking the step to zero. The complete model "
 "has 102,911 parameters, against 37,382 for the direct multi-layer baseline, and its per-step latency on the "
 "test machine is 2.29 ms against 0.20 ms, which matters for the closed-loop results of Section 5.5."),

("h1", "4. Experimental protocol"),
("h2", "4.1. Domains"),
("p",
 "Four domains are used, all implemented from first principles in the same code base and integrated with a "
 "fourth-order Runge-Kutta scheme at a fixed step of 0.02 s. They were chosen to cover the failure modes that "
 "matter for engineering deployment rather than to maximise benchmark coverage: a ballistic vehicle with "
 "attitude dynamics and actuator limits, a multirotor with rotor lag and a quaternion attitude, a strongly "
 "dissipative thermal network, and a reaction-diffusion field with sparse spatial sensing. Table 1 summarises "
 "the state, control, and observation dimensions. A fifth domain is measured rather than simulated: the "
 "run-to-failure turbofan fleet published as the NASA C-MAPSS benchmark. It does not validate the ranking, and "
 "Section 5.9 uses it to bound where the ranking applies: the instrumentation, the noise level and the horizon "
 "of a deployment rather than those of a benchmark."),
("table", "Table 1. Domains, state dimensions, control dimensions, and the number of measured channels.", None,
 [("D1", "Fin-stabilised projectile", "12", "3", "6", "position, attitude"),
  ("D2", "Quadrotor with rotor lag", "17", "4", "7", "position, quaternion"),
  ("D3", "Thermal network, 8 nodes", "8", "3", "8", "all node temperatures"),
  ("D9", "Reaction-diffusion, 24 cells", "72", "2", "24", "every third cell, 3 species"),
  ("D5", "Turbofan fleet, C-MAPSS FD001", "21", "0", "6", "T24, T30, T50, Nf, Ps30, htBleed")]),
("p",
 "The projectile has twelve states, three positions, three velocities, three attitude angles, and three body "
 "rates, with three fin deflections as control. The mass is 8.0 kg, the reference area 0.018 m$^2$, the drag "
 "coefficient 0.22, and the fin lift effectiveness 0.35; the launch speed is sampled uniformly in 180 to 260 m/s "
 "and the flight-path angle in 0.6 to 0.9 rad, controls are Gaussian with standard deviation 0.35, and each "
 "episode lasts 400 steps. The energy functional used for the consistency test is the mechanical energy "
 "consisting of kinetic and gravitational terms."),
("p",
 "The quadrotor has seventeen states, position, velocity, a unit quaternion, body rates, and four rotor speeds "
 "with a first-order lag of 6.0 s$^{-1}$. The mass is 1.2 kg, the arm length 0.25 m, the inertia diagonal is "
 "(0.02, 0.02, 0.04) kg m$^2$, thrust scales with the square of rotor speed with coefficient 2.2, and the "
 "hover input is 1.15 with a standard deviation of 0.06."),
("p",
 "The thermal network is a ring of eight capacitances with conductances between neighbours, three actuated nodes "
 "(0, 3, 6), a 300 K ambient, and a radiative loss; the energy functional is the quadratic deviation from "
 "ambient weighted by the capacitances, which is the quantity whose drift is reported in Section 5.6. The "
 "reaction-diffusion domain has twenty-four cells and three species, with diffusion coefficients 0.28, 0.16, "
 "and 0.10, two actuated regions, and nonlinear reaction terms that produce travelling fronts."),
("p",
 "The measured domain is the FD001 subset of the C-MAPSS turbofan benchmark [29]: one hundred run-to-failure records, "
 "each a sequence of cycles over twenty-one sensor channels. The subset is recorded at a single operating point, "
 "so there is no exogenous input (Table 1 lists a control dimension of zero) and the sequence is driven only by "
 "degradation. Six channels are shown to the model, T24, T30, T50, Nf, Ps30 and htBleed, and the other fifteen "
 "stay hidden, which reproduces the partial-observability protocol of the simulated plants on measured data. "
 "Eighty-four of the hundred records hold the 161 consecutive cycles the protocol requires; sixteen are held "
 "out, sixty-eight train with one window each, and the horizons are 25, 50 and 100 cycles because the record "
 "caps a rollout at 161 cycles."),
("p",
 "One consequence of that window choice deserves stating, because the benchmark is usually scored differently. "
 "Each window is the first 161 cycles of a record, and the shortest held-out record runs to cycle 178, so no "
 "evaluation window contains the failure of its engine; the remaining life at the end of a held-out window ranges "
 "from 17 to 180 cycles. A remaining-life score would require extrapolating past the end of every window, a "
 "regime in which none of the models here is validated. We report error at 25, 50 and 100 cycles instead, all "
 "inside the recorded life of every engine, and no remaining-life metric. Figure 7 shows the measured and "
 "predicted trend of the outlet static pressure, the channel whose drift is monotone in this subset."),
("p",
 "The multi-regime subsets were obtained and are scored only on the degradation target of Section 5.9. Their "
 "operating settings change at almost every cycle, so consecutive samples belong to different regimes; in "
 "normalised coordinates the last-value predictor is off by a factor of the square root of two at one step, so "
 "adjacent readings are effectively independent and a within-record window is not a trajectory a dynamics model "
 "can be scored on. FD001 does not have this artefact, and it is the subset "
 "that the condition-monitoring literature reports most often [39, 40], so the reconstruction "
 "comparison of Section 5.9 is run there and the "
 "limitation is stated rather than hidden behind an aggregate score."),
("h2", "4.2. Partial observability and history window"),
("p",
 "The sensor model is the part of the protocol that matters most for the comparison. On the projectile only "
 "position and attitude are measured; the velocities and body rates are hidden. On the quadrotor only position "
 "and the quaternion are measured, and the body rates and rotor speeds are hidden. On the thermal network all "
 "eight temperatures are measured, which makes it the control case in which the latent model has nothing to "
 "estimate. On the reaction-diffusion field every third cell is measured for all three species, so the "
 "observation is a sparse spatial sample of a seventy-two-dimensional state."),
("p",
 "Every model, including the direct baselines, receives a window of four consecutive observations. The window "
 "is what allows a direct predictor to compete: a multi-layer network applied to four stacked observations can "
 "form finite differences and recover an approximation of the hidden velocities, which is exactly what a "
 "structured model would do with an encoder. Making the window identical across models is therefore essential; "
 "an earlier version of this study, in which the full state was observed and models saw a single frame, "
 "produced the opposite ranking, because the direct model was given a Markovian state while the latent model "
 "was not. Observation noise is 1% of the channel standard deviation, added before the window is formed."),
("h2", "4.3. Baselines"),
("p",
 "Six model families are compared. The direct multi-layer predictor maps the flattened window and the control "
 "to the next observation and maintains a rolling window at test time; it is the reference that any structured "
 "model has to beat. The neural-ODE baseline integrates a learned vector field with a fourth-order Runge-Kutta "
 "solver over the sampling interval on the same rolling window, and is the continuous-time member of the family "
 "introduced by Chen et al. [2]. The gated recurrent latent model carries a "
 "recurrent state and decodes from it, which is the natural non-physical latent baseline. The Hamiltonian "
 "network applies a learned Hamiltonian directly to the measured channels without an encoder, which is the "
 "structural baseline that assumes full observability. The residual latent model has the same encoder, the same "
 "residual decoder, and the same curriculum as PCL-WM but no canonical split and no port-Hamiltonian structure; "
 "it is the control that isolates structure from architecture. PCL-WM is the full model of Section 3."),
("p",
 "The direct predictor is closely related to the dynamics models used for model-based reinforcement learning on "
 "proprioceptive state [15], with the difference that it consumes a window rather than a single frame and is "
 "trained on a reconstruction loss rather than a reward."),
("p",
 "Seven ablations of PCL-WM are trained on the projectile domain: removal of the symplectic step, removal of "
 "the gauge penalty, replacement of the constrained dissipation by a free one, removal of the certificate "
 "regulariser, removal of the curriculum, and two latent widths of 16 and 64. Each ablation changes exactly one "
 "factor relative to the full model, and the training protocol is otherwise identical."),
("h2", "4.4. Metrics"),
("p",
 "Accuracy is the normalised root-mean-square error of the reconstructed observation at horizons of 25, 50, "
 "100, and 200 steps, in normalised coordinates, averaged over sixteen evaluation episodes starting fifty steps "
 "in. Error growth is summarised by fitting a straight line to the logarithm of the "
 "error against the logarithm of the horizon, Eq. (17); the slope $\\beta$ is one for a linearly growing error, "
 "below one for a saturating error, and above one for an error that accelerates. Robustness is measured by the "
 "same metric on a test set whose initial speed is shifted to 240 to 320 m/s, above the training range, with the "
 "model trained on only sixty trajectories. Energy consistency is the mean absolute relative deviation of the "
 "physical energy functional reconstructed from the rollout, in percent, and is reported only for the domain "
 "whose observation covers the full state, because a partially observed domain has no well-defined "
 "reconstructed energy. Closed-loop performance is the root-mean-square position tracking error of a "
 "cross-entropy model predictive controller that plans with the learned model and executes on the exact "
 "simulator, with horizon 15, 128 candidate sequences, 3 iterations, 16 elites, and a control penalty of 0.02."),
("eq", 17),
("p",
 "The controller minimises the cost of Eq. (18) over the planning horizon, with $Q$ weighting position and "
 "attitude and $\\lambda$ the control penalty, and the reported number is the error against the reference "
 "trajectory of the plant, not against the model's own prediction."),
("eq", 18),
("h2", "4.5. Training and hardware"),
("p",
 "All models are trained for 60 epochs with a batch size of 64, the Adam optimiser at a learning rate of "
 "1.5e-3 with cosine annealing, 25 batches per epoch, and a discount of 0.9 per rollout step. The curriculum "
 "reaches a rollout length of 30 at two thirds of training. Training data are 240 trajectories of 400 steps, "
 "except in the data-efficiency study, which uses 60. The evaluation uses 16 held-out trajectories. Experiments "
 "run on a single graphics processor with 24 threads; the structured model trains in 242 s per configuration "
 "and the direct baseline in 36 s. Every configuration in the simulated study is trained with two seeds and the "
 "tables report the mean; where the two runs disagree by more than the difference being claimed, the spread is "
 "reported in Section 6.4 so that the reader can judge "
 "which differences are meaningful."),
]
