# ---- device plan steps ---------------------------------------------------------

abstract type AbstractDevicePlanStep end
abstract type AbstractDeviceChoiceStep <: AbstractDevicePlanStep end

struct DeviceNormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# reparam=:noncentered normal: z read raw from the unconstrained params,
# scored against N(0, 1); the binding materializes theta = mu + sigma * z.
struct DeviceNoncenteredNormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    binding_slot::Int32
end

struct DeviceLognormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceExponentialChoiceStep{R} <: AbstractDeviceChoiceStep
    rate::R
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceGammaChoiceStep{SH,R} <: AbstractDeviceChoiceStep
    shape::SH
    rate::R
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceLaplaceChoiceStep{M,S} <: AbstractDeviceChoiceStep
    loc::M
    scale::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceCauchyChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceHalfNormalChoiceStep{S} <: AbstractDeviceChoiceStep
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceHalfCauchyChoiceStep{S} <: AbstractDeviceChoiceStep
    scale::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceLogisticChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    scale::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceGumbelChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    scale::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceFrechetChoiceStep{SH,SC} <: AbstractDeviceChoiceStep
    shape::SH
    scale::SC
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceRayleighChoiceStep{S} <: AbstractDeviceChoiceStep
    scale::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceInverseGaussianChoiceStep{M,L} <: AbstractDeviceChoiceStep
    mu::M
    lambda::L
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceBetaChoiceStep{A,B} <: AbstractDeviceChoiceStep
    alpha::A
    beta::B
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceStudentTChoiceStep{N,M,S} <: AbstractDeviceChoiceStep
    nu::N
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceInverseGammaChoiceStep{SH,SC} <: AbstractDeviceChoiceStep
    shape::SH
    scale::SC
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceWeibullChoiceStep{SH,SC} <: AbstractDeviceChoiceStep
    shape::SH
    scale::SC
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceBernoulliChoiceStep{P} <: AbstractDeviceChoiceStep
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# Logit-parameterized Bernoulli with a generic scalar `eta` (issues #135/#149):
# `bernoullilogit(scalar_expr)`. Scored in the stable `x*eta - log1p(exp(eta))`
# form. bernoullilogit latents are unsupported (backend rejects them), so
# `value_source` is always the staged observation (-1); the transform field is
# carried for the choice-step convention and stays identity.
struct DeviceBernoulliLogitChoiceStep{E<:AbstractDeviceExpr} <: AbstractDeviceChoiceStep
    eta::E
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# Logit-parameterized Bernoulli whose eta is the fused GLM linear predictor
# `intercept + sum_d coef[d] * X[d, index]` (issues #135/#150 -- the device analog
# of the HOST batched analytic lowering). `D` is the compile-time coefficient
# dimension (the backend guarantees a VectorIdentity latent vector), unrolling the
# dot product in the kernel body. The covariate column `X[:, index]` is
# per-observation CONSTANT data staged into the observation buffer (like observed
# values, addresses erased): staging pushes `D` covariate rows then the observed y
# row, and the kernel reads them cursor-first, so the `index` expr disappears from
# the device plan entirely. `coef_value_source` is the coefficient latent's
# UNCONSTRAINED parameter row start (VectorIdentity: constrained == unconstrained,
# log-abs-det 0); the gradient kernel seeds derivative 1 there so forward-mode
# reproduces `d_eta/d_coef[d] = X[d, index]`, and `d_eta/d_intercept = 1` flows
# through the intercept sub-expr's slot dual. `value_source` is always -1
# (bernoullilogit is observed-only).
struct DeviceBernoulliLogitGLMChoiceStep{D,I<:AbstractDeviceExpr} <: AbstractDeviceChoiceStep
    intercept::I
    coef_value_source::Int32
    value_source::Int32
    binding_slot::Int32
end

DeviceBernoulliLogitGLMChoiceStep{D}(intercept::I, coef_value_source, value_source, binding_slot) where {D,I} =
    DeviceBernoulliLogitGLMChoiceStep{D,I}(
        intercept,
        Int32(coef_value_source),
        Int32(value_source),
        Int32(binding_slot),
    )

# Runtime-length variant of the fused GLM step (issue #221). `coef_length` is a
# runtime `Int32` FIELD rather than a `{D}` type parameter, so the linear
# predictor is scored by a runtime device loop instead of a compile-time-unrolled
# `Val(D)` fold. This is what lifts the D <= 16 coefficient cap for the GLM class
# (the unroll-by-D shader-compile-budget hazard the cap guarded): D <= 16 keeps
# the measured-fast unrolled `{D}` step, D > 16 uses this loop step. Its gradient
# uses the in-kernel ANALYTIC form (`d/d_eta = y - logistic(eta)` fanning to the
# intercept + covariate-scaled coefficient slots), so the gradient kernel also
# avoids the per-coefficient dual arithmetic (and its D-wide unroll).
struct DeviceBernoulliLogitGLMChoiceStepDyn{I<:AbstractDeviceExpr} <: AbstractDeviceChoiceStep
    intercept::I
    coef_value_source::Int32
    coef_length::Int32
    value_source::Int32
    binding_slot::Int32
end

DeviceBernoulliLogitGLMChoiceStepDyn(
    intercept::I,
    coef_value_source,
    coef_length,
    value_source,
    binding_slot,
) where {I} = DeviceBernoulliLogitGLMChoiceStepDyn{I}(
    intercept,
    Int32(coef_value_source),
    Int32(coef_length),
    Int32(value_source),
    Int32(binding_slot),
)

# Broadcast (vectorized) normal observation `{:y} ~ normal.(mu_expr, sigma)` --
# docs/dsl.md's flagship GPU-lowering form (issue #134). A VECTOR observation
# `y[1..M]` scored per element against `normal(mu_i, sigma)`, where `mu_i` is a
# broadcast affine expression over one or more per-observation COVARIATE vectors
# (model arguments) and scalar latents/args, and `sigma` is a scalar expression
# (a latent scale, a scalar arg, or a literal). This is dense per-column scoring
# of a vector observation -- exactly what the device batched logjoint is good at
# (the issue's own framing), so the whole vector reduces in one thread-serial fold
# with no per-element choice steps.
#
# Layout mirrors the GLM step (issue #150): the covariate columns ride the
# observation buffer, addresses erased. Per element `i` staging emits `C` covariate
# rows (one per distinct covariate vector referenced, in the mu-then-sigma
# first-encounter order the lowering assigns offsets) followed by the observed
# `y[i]` row, so one element consumes `C + 1 == stride` observation rows and the
# kernel reads them cursor-first. `mu`/`sigma` are device expressions whose scalar
# leaves read the `slots` buffer (materialized latent/arg bindings) and whose
# covariate leaves are `DeviceObservedColumnExpr(offset)` reading `elem_base +
# offset`. `count_id` indexes the shared `trip_counts` buffer (the broadcast step
# takes a loop-id slot like a `DeviceLoopStep`; staging fills it with `M`), so the
# element count `M` -- resolved once per workspace from the observation vector
# length -- is uniform across the batch, matching the host `_broadcast_uniform_length`.
# `value_source` is always -1 (broadcast latents are rejected at BACKEND lowering);
# the binding slot is carried but NEVER materialized (a vector binding, like the
# mvnormal families) so a downstream read is honestly rejected by the audit.
struct DeviceBroadcastNormalChoiceStep{M<:AbstractDeviceExpr,S<:AbstractDeviceExpr} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    count_id::Int32
    stride::Int32
    value_source::Int32
    binding_slot::Int32
end

struct DeviceBinomialChoiceStep{N,P} <: AbstractDeviceChoiceStep
    trials::N
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceGeometricChoiceStep{P} <: AbstractDeviceChoiceStep
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceNegativeBinomialChoiceStep{R,P} <: AbstractDeviceChoiceStep
    successes::R
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceCategoricalChoiceStep{PS<:Tuple} <: AbstractDeviceChoiceStep
    probabilities::PS
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# Finite mixture of normals (issue #50): the VALUE is a scalar (Identity latent
# or observed), so this follows the group-1 scalar conventions -- the binding
# IS materialized, no audit special-case -- with per-component expression
# tuples and a log-sum-exp fold.
struct DeviceMixtureNormalChoiceStep{W<:Tuple,M<:Tuple,S<:Tuple} <: AbstractDeviceChoiceStep
    weights::W
    mus::M
    sigmas::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DevicePoissonChoiceStep{L} <: AbstractDeviceChoiceStep
    lambda::L
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# Diagonal multivariate normal (issue #12 group 3, phase 1). The dimension is
# compile-time (the argument tuples' arity); a latent reads `D` consecutive
# unconstrained rows starting at `value_source` (VectorIdentity transform:
# constrained == unconstrained, log-abs-det 0), an observation consumes `D`
# staged rows. The binding slot is carried but NEVER written -- the slots
# matrix holds one scalar per symbol -- so the read/write audit treats it as
# unmaterialized and honestly rejects any downstream read.
struct DeviceMvNormalChoiceStep{M<:Tuple,S<:Tuple} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    binding_slot::Int32
end

# Runtime-length diagonal (iid) normal prior over a VectorIdentity latent vector
# (issue #221). The D per-component normal logpdfs are a runtime device loop over
# `dimension` instead of the compile-time `Val(D)` tuple fold in
# DeviceMvNormalChoiceStep, so the kernel body no longer scales with D. Only the
# IID case (every component shares one scalar `mu`/`sigma` expr) lowers here; it is
# what lets a D > 16 GLM's coefficient prior lower (the fused GLM likelihood step
# has its own runtime variant). `value_source` is the latent's unconstrained row
# start; the binding slot is carried but never written (vector bindings are
# unmaterialized, as in DeviceMvNormalChoiceStep).
struct DeviceDiagNormalChoiceStepDyn{M<:AbstractDeviceExpr,S<:AbstractDeviceExpr} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    dimension::Int32
    value_source::Int32
    binding_slot::Int32
end

# Dirichlet (issue #12 group 3, phase 2): the first dimension-changing vector
# step. A latent reads K-1 consecutive unconstrained rows starting at
# `value_source` and constrains them through the register-resident shifted
# softmax (log-abs-det included); an observation consumes K staged rows. The
# binding slot is carried but never written (see DeviceMvNormalChoiceStep).
struct DeviceDirichletChoiceStep{A<:Tuple} <: AbstractDeviceChoiceStep
    alpha::A
    value_source::Int32
    binding_slot::Int32
end

# Dense multivariate normal (issue #12 group 3, phase 3). The constant
# scale_tril factor rides the observation buffer: staging packs its lower
# triangle (column-major) into d(d+1)/2 cursor rows ahead of any observed
# value rows, so the kernel needs no extra buffer or signature change and
# per-batch factors work for free. mu is a compile-time expr tuple; the value
# follows the mvnormal latent/observed conventions.
struct DeviceMvNormalDenseChoiceStep{M<:Tuple} <: AbstractDeviceChoiceStep
    mu::M
    value_source::Int32
    binding_slot::Int32
end

# Diagonal-scale multivariate Student-t: the heavy-tailed `DeviceMvNormalChoiceStep`.
# `nu` is a compile-time scalar arg expression (a latent-dependent df flows in as a
# slot expression); mu/sigma are arg tuples; the value follows the mvnormal
# latent/observed conventions. Binding carried but never written.
struct DeviceMvStudentTChoiceStep{N,M<:Tuple,S<:Tuple} <: AbstractDeviceChoiceStep
    nu::N
    mu::M
    sigma::S
    value_source::Int32
    binding_slot::Int32
end

# Dense-scale multivariate Student-t: the heavy-tailed `DeviceMvNormalDenseChoiceStep`,
# with an added `nu` scalar arg expression. The constant scale_tril factor rides
# the observation buffer exactly as for mvnormaldense.
struct DeviceMvStudentTDenseChoiceStep{N,M<:Tuple} <: AbstractDeviceChoiceStep
    nu::N
    mu::M
    value_source::Int32
    binding_slot::Int32
end

# LKJ prior over a packed correlation Cholesky factor (issue #57, mirroring
# the CPU-native step from issue #49). `D` is the compile-time dimension (the
# backend step guarantees a macro-time literal); a latent reads d(d-1)/2
# unconstrained rows starting at `value_source` and constrains them through
# the register-resident tanh/stick construction (log-abs-det included), an
# observation consumes d(d+1)/2 staged packed rows. The binding slot is
# carried but never written (see DeviceMvNormalChoiceStep).
struct DeviceLKJCholeskyChoiceStep{D,E} <: AbstractDeviceChoiceStep
    eta::E
    value_source::Int32
    binding_slot::Int32
end

DeviceLKJCholeskyChoiceStep{D}(eta::E, value_source, binding_slot) where {D,E} =
    DeviceLKJCholeskyChoiceStep{D,E}(eta, Int32(value_source), Int32(binding_slot))

# Compile-time-unrolled tuple folds multiply the fused kernel body by D (and
# the gradient kernel again by parameter count); cap the dimension so Metal
# shader compilation stays inside its budget (see docs/device-vector-latents.md).
const DEVICE_MAX_VECTOR_DIMENSION = 16

# The dense forward substitution unrolls d(d+1)/2 fused multiply-adds per step
# (doubled again in the gradient kernel), so its cap is tighter.
const DEVICE_MAX_DENSE_DIMENSION = 8

# The unrolled cholesky-correlation constrain and row-norm support checks are
# likewise quadratic in the kernel body, so lkjcholesky shares the dense cap.
const DEVICE_MAX_CHOLESKY_DIMENSION = 8

# Truncated families are observed-only on the backend path (latents fall back at
# backend lowering: the bounded transform is unimplemented there), so the value
# source is always the staged observation; the fields stay generic regardless.
struct DeviceTruncatedNormalChoiceStep{M,S,L,U} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    lower::L
    upper::U
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceTruncatedStudentTChoiceStep{N,M,S,L,U} <: AbstractDeviceChoiceStep
    nu::N
    mu::M
    sigma::S
    lower::L
    upper::U
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceDeterministicStep{E} <: AbstractDevicePlanStep
    expr::E
    binding_slot::Int32
end

struct DeviceLoopStep{B<:Tuple} <: AbstractDevicePlanStep
    loop_id::Int32
    iterator_slot::Int32
    body::B
end

# marginalize=:enumerate (issue #67, the device mirror of the CPU-native
# BackendMarginalizeChoicePlanStep). A finite-support discrete latent summed out
# analytically: the step OWNS its plan suffix as a nested lowered step tuple (the
# `DeviceLoopStep` body precedent) and folds the per-support-value branch totals
# through a max-shifted log-sum-exp -- exactly the CPU right-fold semantics
# (docs/discrete-enumeration.md), using the mixture-normal fold as the in-kernel
# template. `F` is the family (`:bernoulli` / `:categorical`, a Symbol type param)
# selecting the per-branch pmf; `support` is the compile-time tuple of the BINDING
# values the branches materialize (bernoulli: `(0, 1)`; categorical: `(1..K)`),
# bounded by the backend's 32 total-support-product cap so the unrolled kernel body
# stays inside the Metal shader budget. `probabilities` holds the lowered pmf
# argument exprs (bernoulli: `(p,)`; categorical: the K entries).
#
# The step consumes ONE staged SELECTOR observation row (0 == marginalize over all
# branches; `v` in `1..S` == conditioned on branch `v`; anything else == -Inf) so
# the observation cursor stays aligned across the batch; every branch then re-scores
# the identical suffix from the SAME post-selector cursor and advances it equally
# (issue #67 point 4). The binding slot IS materialized per branch (a scalar, unlike
# the vector families) so the suffix can read it; the per-branch value carries a
# ZERO derivative in the gradient kernel (a discrete branch constant), reproducing
# the CPU cleared-slot-gradient contract.
struct DeviceMarginalizeChoiceStep{F,PS<:Tuple,SP<:Tuple,B<:Tuple} <: AbstractDeviceChoiceStep
    probabilities::PS
    support::SP
    body::B
    binding_slot::Int32
end

DeviceMarginalizeChoiceStep{F}(probabilities::PS, support::SP, body::B, binding_slot) where {F,PS<:Tuple,SP<:Tuple,B<:Tuple} =
    DeviceMarginalizeChoiceStep{F,PS,SP,B}(probabilities, support, body, Int32(binding_slot))

struct DeviceExecutionPlan{T,S<:Tuple}
    steps::S
    slot_count::Int32
    loop_count::Int32
end

DeviceExecutionPlan{T}(steps::S, slot_count, loop_count) where {T,S<:Tuple} =
    DeviceExecutionPlan{T,S}(steps, Int32(slot_count), Int32(loop_count))

function Base.show(io::IO, plan::DeviceExecutionPlan{T}) where {T}
    print(
        io,
        "DeviceExecutionPlan{",
        T,
        "}(steps=",
        length(plan.steps),
        ", slots=",
        plan.slot_count,
        ", loops=",
        plan.loop_count,
        ")",
    )
end
