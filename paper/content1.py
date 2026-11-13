# -*- coding: utf-8 -*-
"""Front matter, Introduction and Related work. Blocks are consumed by build.py."""

BLOCKS = [
("h1", "Highlights"),
("bullet", "Physics priors are scored component by component in a latent world model."),
("bullet", "Hard symplectic steps cut long-horizon accuracy by 2.3x in latent space."),
("bullet", "Port-Hamiltonian dissipation with PSD R=LLT is the prior that transfers."),
("bullet", "With 60 trajectories the structured model is about 15% more accurate."),
("bullet", "Closed-loop MPC hides model error that open-loop rollout exposes."),

("h1", "Abstract"),
("p",
 "Latent world models replace an expensive simulator with a differentiable map, and physics-informed variants are usually reported as uniformly better. This paper asks which of those priors actually help, as separable design decisions. The model studied splits a latent state into a canonical pair advanced by a learned Hamiltonian through a Störmer-Verlet map and a residual block following discrete port-Hamiltonian dynamics with a positive-semidefinite dissipation matrix, plus a gauge condition and an online error certificate. Sixty-three trained configurations on a projectile, a quadrotor, a thermal network, a reaction-diffusion field and the measured C-MAPSS fleet, under partial observability, give a conditional picture. On the projectile the Verlet map costs 2.3 times in 200-step error, 0.793 against 0.343, growth exponent 1.33 to 0.96; on the quadrotor the same removal buys 1.37 times, on the thermal network 1.09, and on the reaction-diffusion field it costs 2.6 times, because the implicit stage damps stiff latent modes. The integrator is a spectrum-matching decision. What survives is the port-Hamiltonian residual with constrained dissipation and the gauge condition, and only out of distribution: 0.595 against 0.704 and 0.744 with sixty trajectories and initial speeds shifted outside the training range, about 16% better at 9% cost in distribution. Closed-loop control separates none of them, 0.065 to 0.074. On the measured fleet true-equation estimators stay two to a hundred times more accurate, fixing how much of that gap has closed. The AI contribution is the component-wise attribution; the engineering one, a costed account of which priors to keep."),

("p", "Keywords: world models; latent dynamics; port-Hamiltonian systems; symplectic integration; "
      "model predictive control; partial observability; engineering applications"),

("h1", "Nomenclature"),
("nomen", [
 ("$z_k$", "latent state at step k"),
 ("$q_k,\\ p_k$", "canonical position and momentum blocks"),
 ("$\\xi_k$", "residual latent block"),
 ("$m$", "number of canonical degrees of freedom"),
 ("$r$", "dimension of the residual block"),
 ("$H_\\theta$", "learned Hamiltonian"),
 ("$H_r$", "learned residual energy function"),
 ("$J$", "canonical symplectic matrix"),
 ("$J_r$", "skew-symmetric residual interconnection"),
 ("$R$", "dissipation matrix, $R=LL^{\\top}$"),
 ("$L$", "lower-triangular factor of the dissipation"),
 ("$G$", "input map of the residual block"),
 ("$h$", "learned integration step"),
 ("$u_k$", "control input at step k"),
 ("$o_k$", "measured observation vector"),
 ("$\\hat{o}_k$", "reconstructed observation"),
 ("$E_\\phi,\\ D_\\psi$", "encoder and residual decoder"),
 ("$L$", "observation history length"),
 ("$K$", "training rollout length"),
 ("$\\beta$", "error-growth exponent"),
 ("$N_{\\mathrm{cert}}$", "certified prediction horizon"),
]),

("h1", "1. Introduction"),
("p",
 "Learning a model of a physical plant from logged trajectories has become a practical alternative to "
 "first-principles modelling whenever the plant is too complex, too poorly characterised, or too expensive to "
 "simulate at the rate a controller needs. The dominant recipe is a world model: an encoder compresses "
 "observations into a latent state, a transition map advances that state, and a decoder turns the predicted "
 "state back into quantities a controller can use. The recipe has been very successful in reinforcement "
 "learning, where the model only has to be accurate enough to rank candidate action sequences, and it is now "
 "being imported into engineering control, where the model also has to survive long rollouts, sparse sensing, "
 "and operating points that were not in the training set."),
("p",
 "A natural way to make such models trustworthy is to give them physical structure. The most popular choice "
 "borrows classical mechanics: split the latent state into positions and momenta, learn a scalar Hamiltonian, "
 "and advance the state with a symplectic integrator, so that the discrete map inherits the conservation "
 "properties of the continuous flow. Variants replace the Hamiltonian by a Lagrangian, add a dissipation term, "
 "or impose port-Hamiltonian form with an interconnection and a damping matrix. The claim attached to these "
 "constructions is usually the same: better long-horizon accuracy, better data efficiency, better extrapolation. "
 "That claim is supported in the papers that introduce them, almost always on benchmarks where the full state "
 "is observed, the system is conservative or lightly damped, and the evaluation horizon is short relative to "
 "the training rollout."),
("p",
 "This paper started from the opposite observation. When the same structured model was placed in an engineering "
 "setting, with only the measured channels available, a four-step history window, and rollouts of two hundred "
 "steps, it did not merely fail to beat the baselines; it was the worst model in the comparison, and the "
 "component that was supposed to help, the symplectic step, turned out to be the component that hurt. Rather "
 "than tune the construction until the original claim could be recovered, we treated the result as the object "
 "of study. Structure was decomposed into its parts, each part was removed in turn under an otherwise identical "
 "training protocol, every configuration was trained twice, and the resulting grid was compared on accuracy, error growth, "
 "robustness to distribution shift, and closed-loop control performance, with the last group of runs executed on "
 "a public measured fleet so that the conclusions are not conditional on our own simulator."),
("p",
 "The decomposition gives a conclusion that is more useful than a single aggregate score. The symplectic step, "
 "implemented as a fixed-step Störmer-Verlet map inside a learned latent space, is the largest single error "
 "source on the plant where the canonical split is closest to the physics: on the projectile domain it raises the "
 "200-step error from 0.343 to 0.793 and the error-growth exponent from 0.96 to 1.33. The same removal buys much "
 "less elsewhere, 1.37 times on the quadrotor and 1.09 times on the thermal network, and it costs 2.6 times on the "
 "reaction-diffusion field, where the implicit stage of the map damps stiff latent modes that a removed map "
 "exposes. The port-Hamiltonian residual block with a positive-semidefinite dissipation matrix, together with the "
 "gauge condition that removes the translation symmetry of the canonical block, is the part that generalises. Its "
 "advantage is invisible in distribution and clear out of it: with sixty training trajectories and initial speeds "
 "shifted above the training range, the structured model reaches 0.595 against 0.704 for an unstructured latent "
 "model and 0.744 for a direct predictor, on two seeds per configuration. In distribution that group is nine "
 "percent behind the strongest unstructured baseline, so the claim is confined to the shifted regime, and we do "
 "not attempt to separate the constraints inside the group."),
("p",
 "We also report what does not work, because it constrains how the result should be read. On "
 "the thermal network, where every state channel is measured and the dynamics are strongly "
 "dissipative, the direct multi-layer predictor is two orders of magnitude more accurate "
 "than any latent model, and its relative energy deviation over two hundred steps is 1.63% "
 "against 355% for the structured model. On the reaction-diffusion field, where the "
 "observation is a sparse spatial sample, the direct predictor is again the most accurate. A "
 "residual latent model with no physical structure at all matches the best structured "
 "configuration in distribution, which says that the gains usually attributed to geometry "
 "are, in this regime, produced by the residual decoding and the long-horizon training "
 "curriculum. In closed loop the picture reverses in a different sense: the differences are "
 "absorbed rather than amplified. Repeated over three controller runs and twenty-four "
 "episodes, every configuration tracks with a normalised position error between 0.0649 and "
 "0.0739, and the run-to-run spread of a single configuration reaches 0.015, so the closed "
 "loop cannot be used to rank these models. It does establish that feedback compensates the "
 "open-loop deficiencies the tables measure, which argues for a short closed-loop trial "
 "rather than open-loop accuracy alone."),
("p",
 "The final group of runs leaves our own simulator. The protocol is executed unchanged on the NASA C-MAPSS FD001 "
 "turbofan fleet, sixty-eight training records and sixteen held-out records of measured sensor channels, and the "
 "result is again not uniform. Every model beats the last-value predictor, the best families by 22% at 25 cycles and "
 "39% at 100 cycles, but the error curve flattens instead of growing (exponent 0.02 against 0.9 to 1.0 in the "
 "simulator) because the records are noise limited at the cycle scale, and the structured model is not the most "
 "accurate predictor on them, the direct predictor is. What the fleet does confirm is the boundedness argument: "
 "removing the Verlet map takes the 100-cycle error from 0.971 to 14.484 with all four seeds diverging, "
 "whereas the structured model stays between 0.743 and 1.890 and leaves the fitted region in two runs out of "
 "six. Structure therefore buys a "
 "bounded rollout rather than a lower error on measured data, and the two effects are reported separately."),
("p",
 "The contributions of the paper are therefore stated as four separable claims rather than one."),
("p",
 "First, we give an evaluation protocol for latent world models in engineering control that removes the "
 "confound that hides most of the interesting behaviour: partial observability. All models see only the "
 "measured channels, all models receive the same history window, all are trained with the same rollout "
 "curriculum, and all are scored on the same normalised horizons. Under this protocol the direct baselines "
 "degrade by an order of magnitude relative to a fully observed version of the same experiment, which is the "
 "condition under which a state estimator, and therefore a latent model, has something to contribute."),
("p",
 "Second, we separate the cost of the discrete integrator from the cost of the representation. The same ablation "
 "is repeated on all four plants. It gains a factor of 2.3 on the projectile, 1.37 on the quadrotor, and 1.09 on "
 "the thermal network, and it loses a factor of 2.6 on the reaction-diffusion field. That reversal is the finding: "
 "a hard symplectic step is a net cost where the learned Hamiltonian approximates a near-conservative flow and the "
 "latent coordinates are close to canonical, and a net benefit where the latent spectrum contains stiff modes that "
 "the implicit stage damps. We give three mechanisms for the first case and turn the pair of results into a design "
 "rule that can be checked on a short validation rollout before deployment."),
("p",
 "Third, we isolate the components that do survive: a residual block in port-Hamiltonian form whose dissipation "
 "matrix is parameterised as R=LLT and therefore positive semidefinite by construction, a gauge condition on the "
 "canonical block, and a certificate-based regulariser. These components are cheap and they do not change the "
 "short-horizon accuracy. Their value appears only in the regime that matters for engineering deployment, "
 "namely few trajectories and shifted operating points, where the configuration that keeps them reaches 0.595 "
 "against 0.704 for an unstructured latent model on two seeds; and it appears as a property of the growth law, "
 "with an exponent of 0.947 against 1.018, rather than as a constant offset. The online certificate, which "
 "decomposes the rollout error into five attributable terms and returns a horizon beyond which the prediction "
 "is no longer certified, is retained as a diagnostic; we are explicit that its numeric values in this paper "
 "are derived from the measured decomposition and not from a validated a-priori bound. The decomposition is "
 "condensed into a selection table, Section 6.5 and Table 12, which maps four properties of a plant that can "
 "be read off its instrumentation and its data set onto the priors this study supports and the measurement "
 "each of them rests on."),
("p",
 "Fourth, we bound the three claims from outside the simulator and from above. The protocol is run unchanged on "
 "measured turbofan data, and a pair of state estimators is run with the true plant equations so that the size of "
 "the remaining gap can be stated per plant rather than assumed. That reference is better than every learned family "
 "by a factor of two to two orders of magnitude on three of the four simulated plants, and it loses at the shortest "
 "horizon on one, which locates both the price of not knowing the equations and the part of that price the learned "
 "models have recovered. No claim in the paper survives a comparison in which the equations are available, and "
 "Section 6 says so."),
("p",
 "The remainder of the paper is organised as follows. Section 2 positions the work with respect to latent world "
 "models, structured physical priors, structure-preserving integrators, and learning-based model predictive "
 "control. Section 3 describes the model, the gauge condition, the certificate, and the training objective. "
 "Section 4 defines the four simulated domains and the measured fleet, the sensor model, the baselines, and the "
 "metrics. Section 5 reports the results, table by table. Section 6 discusses why the symplectic step fails, what "
 "should be kept, what the study cannot conclude, and closes in Section 6.5 with the selection table that turns "
 "the decomposition into a decision about a plant. Section 7 concludes."),

("h1", "2. Related work"),
("h2", "2.1. Latent world models"),
("p",
 "The idea of learning dynamics in a compressed latent space and planning inside it goes back to the deep "
 "planning network, which introduced a recurrent state-space model with deterministic and stochastic paths and "
 "used it for online planning from pixels [8]. The line continues with a compact recurrent world model whose "
 "policy was trained entirely inside its own hallucinated rollouts and then transferred back to the environment "
 "[30], and with continuous-time variants that replace the discrete recurrence by an ordinary differential "
 "equation, latent ODEs [31] or a controlled differential equation driven by the observations themselves [32]. "
 "The same architecture, scaled and stabilised, underlies the "
 "current generation of general agents [9], and a temporal-difference variant has shown that a latent model "
 "trained with a planning-consistent objective can be robust across more than a hundred control tasks with one "
 "set of hyperparameters [10]. These results establish that latent rollouts are accurate enough to rank actions. "
 "They do not establish that a latent model is a good predictor of measured engineering quantities over long "
 "horizons, because the evaluation is usually a task return rather than a trajectory error, and because the "
 "agent is allowed to re-plan every few steps, which masks compounding error."),
("p",
 "Probabilistic ensembles and trajectory sampling address compounding error by propagating uncertainty [3], and "
 "in the continuous-control literature they remain the strongest purely data-driven baseline. Our direct "
 "multi-layer predictor plays the same role here: it is deliberately simple, trained on the same windows, and "
 "serves as the reference that a structured latent model has to beat before its extra machinery can be "
 "justified."),
("p",
 "The same latent machinery has recently been combined with physical structure. Port-Hamiltonian generative "
 "transitions have been embedded inside a recurrent world model and used for energy-guided policy learning on "
 "visual control benchmarks [25], and a soft-Hamiltonian model splits the latent into a canonical pair and a "
 "context block, conditions it on a selective state-space memory, and plans with the cross-entropy method [26]. "
 "Both report that physical organisation improves imagined rollouts, and both evaluate on standard visual control "
 "suites rather than on instrumented engineering plants. Section 2.5 sets out the differences in detail."),
("h2", "2.2. Structured physical priors"),
("p",
 "Injecting mechanics into a network has a long record. Hamiltonian neural networks learn a scalar function "
 "whose symplectic gradient reproduces the observed vector field, and conserve energy by construction [5]. "
 "Lagrangian variants learn the mass matrix and the potential and obtain the equations of motion by "
 "differentiation [13], and controlled extensions add a port so that inputs enter through a learned input "
 "matrix [22]. Dissipative variants replace the skew-symmetric interconnection by a port-Hamiltonian pair "
 "(J-R), with R symmetric positive semidefinite, so that the energy derivative is bounded by the supply rate "
 "[23]. Generative versions learn a Hamiltonian in a latent space from images and can roll out forward and "
 "backward in time [20]."),
("p",
 "Two other families are relevant. Graph-based simulators learn particle or mesh dynamics with message passing "
 "and generalise across resolutions and initial conditions [16, 19], which is the most convincing evidence that "
 "structural inductive bias buys extrapolation. Operator learning targets the solution map of a partial "
 "differential equation directly and is discretisation invariant [12]. Linear-embedding approaches occupy a "
 "third position between them: a Koopman autoencoder learns coordinates in which the dynamics are affine, so "
 "the latent propagator is a matrix exponential and prediction is global rather than local [33], and the same "
 "construction can be combined with sparse identification to recover both the coordinates and the equations "
 "[34] or distilled into symbolic expressions that extrapolate better than the network they came from [35]. "
 "Both are stronger than a small latent "
 "model on the problems they are designed for, and both assume that the state is observable at training time, "
 "which is precisely the assumption we remove."),
("p",
 "Within the symmetry-preserving branch, the closest relative is the family of symplectic networks, which "
 "compose linear, activation and gradient modules so that the resulting map is exactly symplectic and admit "
 "universal approximation over symplectic maps [36]. We use a different composition, a learned Hamiltonian "
 "evaluated through the Verlet map of Eqs. (3) and (4), because the ablation in Section 5.2 is about the "
 "discretisation rather than about the parameterisation; the finding that the map can cost accuracy where the "
 "canonical split is only approximate is, however, directly relevant to that family as well."),
("p",
 "A parallel line discovers equations rather than embedding them: sparse regression over a library of candidate "
 "terms recovers interpretable governing equations from noisy data [1], and physics-informed training adds "
 "residual penalties on the differential equation [11, 17]. The penalty approach is cheap and flexible, and its "
 "weakness is that a soft penalty does not survive a long rollout: the constraint is satisfied on average over "
 "the training distribution, not along every trajectory. The port-Hamiltonian construction used here is the "
 "hard-constrained alternative for the dissipative part, and the ablation in Section 5.3 measures what it is "
 "worth."),
("p",
 "The construction used here is closest to two recent lines. Latent port-Hamiltonian identification has been "
 "formulated as a reduction-plus-identification problem in which an autoencoder maps state data to a low "
 "dimension and the port-Hamiltonian matrices are parameterised through a Cholesky factor so that passivity holds "
 "by construction [24]; this is the source of the factorisation we use for the dissipation matrix. A two-stage "
 "variant learns a contrastive teacher from observation histories and then projects it onto a port-Hamiltonian "
 "realisation, with sufficient conditions for recovering the state up to a diffeomorphism [27]. Those results "
 "concern identifiability of the latent coordinates. Our question is different: given the same construction, does "
 "the structure pay off in the quantities an engineering user cares about, and where does it stop paying."),
("h2", "2.3. Structure-preserving integration"),
("p",
 "Symplectic integrators are the standard answer to long-time integration of Hamiltonian systems: a symmetric "
 "composition such as the Störmer-Verlet map preserves the symplectic form exactly and keeps the energy error "
 "bounded over exponentially long times, with a modified Hamiltonian that the numerical flow follows [6, 7]. "
 "The result is asymptotic in the step size h and requires that the vector field be the gradient of a fixed "
 "Hamiltonian evaluated at the state actually integrated."),
("p",
 "Learning-based work imports the integrator wholesale: the network supplies the Hamiltonian, the integrator "
 "supplies the map, and the composition is reported to inherit the conservation property [22, 4]. What the "
 "import does not carry over is the premise. In a learned latent space the Hamiltonian is re-estimated from "
 "finite data, the latent coordinates are identified only up to a gauge, the effective step is set by the data "
 "sampling rate rather than by stability, and the map is trained against a one-step or few-step loss rather than "
 "against the flow it is supposed to approximate. Section 6 analyses these four mismatches and connects them to "
 "the measured failure."),
("h2", "2.4. Learning-based model predictive control"),
("p",
 "Model predictive control is the natural consumer of a learned model: the controller solves a finite-horizon "
 "optimisation at every step, so model bias enters as a bounded disturbance rather than as an open-loop drift "
 "[14]. Sampling-based planners such as the cross-entropy method are popular when the model is differentiable "
 "but the cost is non-convex, and they tolerate a stochastic model [3]; the path-integral formulation is the "
 "closest relative of the planner we use, and it differs only in how the sampled trajectories are reweighted "
 "[38]. Deployment on real hardware requires "
 "that the model and the optimiser fit a real-time budget, which is why reduced-order and neural models have "
 "been integrated into embedded MPC for quadrotors [18] and why model latency, not only model accuracy, appears "
 "in our cost table."),
("p",
 "The gap this paper addresses is the absence of a controlled comparison in which the same structured model is "
 "scored on open-loop accuracy, error growth, out-of-distribution robustness, and closed-loop tracking under one "
 "protocol. Published comparisons tend to vary the model and the protocol at the same time, which makes the "
 "attribution of a gain to a structural component impossible. We vary only the model, ablate each structural "
 "component in turn, and report the closed-loop consequence."),
("h2", "2.5. Relation to concurrent structured world models"),
("p",
 "Five recent works overlap with ours closely enough that the differences should be stated plainly rather than "
 "left to the reader. Latent port-Hamiltonian identification with a Cholesky-parameterised dissipation [24] "
 "establishes that a passivity-preserving latent model can be fitted to state data; its examples are a "
 "mass-spring-damper chain, a pendulum, and a disc brake, all fully observed. The two-stage contrastive scheme "
 "[27] adds observation histories and identifies the latent state up to a diffeomorphism, with theory rather "
 "than an engineering evaluation as the deliverable. Port-Hamiltonian world models for control [25] and the "
 "soft-Hamiltonian world model with selective memory [26] embed comparable structure in a recurrent world "
 "model, plan with it, and report returns on the DeepMind Control Suite; the position paper [28] argues the "
 "case for the family without experiments. Two further concurrent lines are closer to our second and third "
 "contributions than anything above, and they should be separated explicitly. A port-Hamiltonian forecaster "
 "identified from position-only observations [42] treats the same observation setting as this paper, a strict "
 "subset of the state, and advances the vector field with a Strang splitting; across thirteen forecasting "
 "benchmarks it reports the best long-horizon accuracy among its baselines and separates forecasting stability "
 "from identifiability, arguing that parameter recovery is ill posed without anchors. Its conclusion is the one "
 "we reach from a different split: structure buys stability, which is not the same as buying error. A symplectic "
 "latent world model [43] replaces the predictor of a joint-embedding architecture by an exactly symplectic "
 "leapfrog step over a learned position and momentum pair, and shows that volume preservation rules out "
 "contraction, stable fixed points, and merging of latent states independently of the learned parameters. That "
 "is the boundedness result of Section 5.9 stated as a guarantee, and it is stronger than our empirical version. "
 "Our result contributes the converse case, which their guarantee does not cover: when the latent coordinates "
 "are re-estimated at every rollout step and the canonical assumptions hold only approximately, the same map "
 "costs accuracy, by a factor of 2.6 on the reaction-diffusion field. Three differences therefore separate our "
 "submission from this group. First, "
 "the input here is a fixed, sparse instrumentation set rather than images or a full state, and the evaluation "
 "is trajectory error, error-growth exponent, and closed-loop tracking of a physical quantity, not task return. "
 "Second, we treat the choice of the canonical update as a variable and report that the discrete symplectic "
 "composition, which the works above either assume or inherit, costs accuracy once the Hamiltonian is "
 "re-estimated and the step size is data-driven; we then keep the part of the structure that survives. Third, "
 "the contribution is the decomposition of where a structural prior pays and where it does not, supported by a "
 "grid of trained configurations across four plants and by an external check on measured data, which none of "
 "the works above provides. Every component ablation is run with two seeds, and the two runs are reported "
 "separately where they disagree, because the first seed of the gauge and dissipation configurations "
 "overstated what those constraints cost."),
]
