# Consistency of the single-source distribution family table (issue #331,
# stage 1). The hand-maintained allowlists are derived from
# src/distributions/family_table.jl; this file (a) pins today's memberships
# so accidental drift in a derived list fails HERE with the family named,
# and (b) asserts the coherence invariants between the per-row flags that
# the derivations rely on.

@testset "family table consistency" begin
    famtab = UncertainTea.DISTRIBUTION_FAMILY_TABLE

    # -- pinned memberships (the pre-#331 hardcoded lists, bit-for-bit) -----

    famtab_builtin = Set([
        :normal, :lognormal, :laplace, :exponential, :gamma, :inversegamma,
        :weibull, :beta, :dirichlet, :bernoulli, :bernoullilogit, :binomial,
        :betabinomial, :multinomial, :discreteuniform, :geometric,
        :negativebinomial, :poisson, :studentt, :categorical, :mvnormal,
        :mvnormaldense, :gaussianprocess, :sparsegaussianprocess, :hmm,
        :orderedlogistic, :zeroinflatedpoisson, :zeroinflatednegativebinomial,
        :vonmises, :mvstudentt, :mvstudenttdense, :wishart, :inversewishart,
        :lkjcholesky, :truncatednormal, :truncatedstudentt, :mixture, :iid,
        :cauchy, :halfnormal, :halfcauchy, :uniform, :logistic, :gumbel,
        :pareto, :frechet, :rayleigh, :inversegaussian,
    ])
    @test length(famtab) == length(famtab_builtin)
    @test allunique(spec.family for spec in famtab)
    @test Set(UncertainTea.BUILTIN_DISTRIBUTION_FAMILIES) == famtab_builtin

    # families the macro recognizes as distribution constructors
    famtab_known = setdiff(
        famtab_builtin,
        Set([
            :gaussianprocess, :sparsegaussianprocess, :hmm, :orderedlogistic,
            :zeroinflatedpoisson, :zeroinflatednegativebinomial, :vonmises, :iid,
        ]),
    )
    @test Set(UncertainTea._KNOWN_DISTRIBUTION_FAMILIES) == famtab_known

    # families whose bare name `@tea` rewrites to the package-qualified
    # constructor: the known families plus the iid combinator
    @test Set(UncertainTea._MACRO_QUALIFIED_FAMILIES) == union(famtab_known, Set([:iid]))

    # dot-call broadcast observation families
    @test Set(UncertainTea._BROADCAST_DISTRIBUTION_FAMILIES) ==
          Set([:normal, :poisson, :bernoulli, :bernoullilogit, :exponential, :studentt])

    # scalar latents with an unconditional parameter slot, by transform kind
    @test Set(UncertainTea._IDENTITY_TRANSFORM_LATENT_FAMILIES) ==
          Set([:normal, :laplace, :studentt, :cauchy, :logistic, :gumbel])
    @test Set(UncertainTea._LOG_TRANSFORM_LATENT_FAMILIES) == Set([
        :lognormal, :exponential, :gamma, :inversegamma, :weibull,
        :halfnormal, :halfcauchy, :frechet, :rayleigh, :inversegaussian,
    ])
    @test Set(UncertainTea._LOGIT_TRANSFORM_LATENT_FAMILIES) == Set([:beta])
    @test Set(UncertainTea._SCALAR_TRANSFORM_LATENT_FAMILIES) == union(
        Set(UncertainTea._IDENTITY_TRANSFORM_LATENT_FAMILIES),
        Set(UncertainTea._LOG_TRANSFORM_LATENT_FAMILIES),
        Set(UncertainTea._LOGIT_TRANSFORM_LATENT_FAMILIES),
    )

    # conditional latents (per-family static-argument checks stay hand-written)
    @test Set(spec.family for spec in famtab if spec.latent === :conditional) == Set([
        :uniform, :pareto, :dirichlet, :mvnormal, :mvnormaldense, :mvstudentt,
        :mvstudenttdense, :wishart, :inversewishart, :lkjcholesky,
        :truncatednormal, :truncatedstudentt, :mixture,
    ])

    # latent mixture component eligibility (real-line location-scale)
    @test Set(UncertainTea._MIXTURE_REAL_LINE_FAMILIES) == Set([:normal, :laplace, :studentt])

    # backend-lowerable families: ORDER is pinned too, because the gradient
    # crosscheck suite derives per-family RNG seeds from `enumerate` over it
    @test UncertainTea.GPU_BACKEND_SUPPORTED_DISTRIBUTIONS == [
        :normal, :lognormal, :laplace, :exponential, :gamma, :inversegamma,
        :weibull, :beta, :dirichlet, :bernoulli, :bernoullilogit, :binomial,
        :betabinomial, :discreteuniform, :geometric, :negativebinomial,
        :poisson, :studentt, :categorical, :mvnormal, :truncatednormal,
        :truncatedstudentt, :mixture, :mvnormaldense, :mvstudentt,
        :mvstudenttdense, :lkjcholesky, :cauchy, :halfnormal, :halfcauchy,
        :uniform, :logistic, :gumbel, :pareto, :frechet, :rayleigh,
        :inversegaussian,
    ]

    # device-lowerable families (recorded in the table; the actual gate is
    # `_lower_device_step!` dispatch in device/plan.jl until #331 stage 2
    # generates those wrappers from the table): backend minus the four
    # families without device wrappers
    famtab_device = Set(spec.family for spec in famtab if spec.device)
    @test famtab_device == setdiff(
        Set(UncertainTea.GPU_BACKEND_SUPPORTED_DISTRIBUTIONS),
        Set([:uniform, :pareto, :betabinomial, :discreteuniform]),
    )

    # -- per-row flag coherence ---------------------------------------------

    famtab_scalar_kinds = (:identity, :log, :logit)
    famtab_conditional_kinds = (
        :bounded, :lowerbounded, :simplex, :vector,
        :choleskycov, :choleskycorr, :truncated, :mixture,
    )
    for spec in famtab
        # broadcastable and backend-lowerable families are macro-known
        @test !spec.broadcastable || spec.known
        @test !spec.backend || spec.known
        # every known family is macro-qualified (iid is qualified-only)
        @test !spec.known || spec.macro_qualified
        # device lowering consumes backend plan steps
        @test !spec.device || spec.backend
        # latent kind and transform kind agree
        @test spec.latent in (:none, :scalar, :conditional)
        if spec.latent === :none
            @test spec.transform === nothing
        elseif spec.latent === :scalar
            @test spec.transform in famtab_scalar_kinds
        else
            @test spec.transform in famtab_conditional_kinds
        end
        # only known families carry latent parameter slots
        @test spec.latent === :none || spec.known
        # mixture components are identity-transform scalar latents
        @test !spec.mixture_component ||
              (spec.latent === :scalar && spec.transform === :identity)
    end
end
