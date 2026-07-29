# Public, allocating convenience entry point for the device logjoint path.

"""
    device_batched_logjoint(model, params, args=(), constraints=choicemap();
                            backend=KernelAbstractions.CPU(),
                            precision=float(eltype(params))) -> Vector

Device-resident batched (unconstrained) logjoint via KernelAbstractions. `params` is
a `parameter_count × batch` matrix of unconstrained parameters; the returned vector
holds the per-column unconstrained logjoint (prior + likelihood + transform
log-abs-det), matching `batched_logjoint_unconstrained` on the CPU.

Constructs a fresh `DeviceBatchedWorkspace` each call; use `DeviceBatchedWorkspace` +
`device_batched_logjoint!` to reuse device buffers and staged observations. Throws an
`ArgumentError` (pointing at `device_lowering_report`) if the model is not
representable on the device.
"""
function device_batched_logjoint(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap();
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=float(eltype(params)),
)
    workspace = DeviceBatchedWorkspace(
        model,
        size(params, 2);
        backend=backend,
        precision=precision,
        args=args,
        constraints=constraints,
    )
    return device_batched_logjoint!(workspace, params)
end

"""
    device_batched_logjoint_gradient(model, params, args=(), constraints=choicemap();
                                     backend=KernelAbstractions.CPU(),
                                     precision=float(eltype(params)))
        -> (values::Vector, gradients::Matrix)

Device-resident batched value AND gradient of the UNCONSTRAINED logjoint via a single
fused KernelAbstractions kernel that forward-differentiates the plan in `DeviceDual`
numbers. `params` is a `parameter_count x batch` matrix of unconstrained parameters.
`values` is the per-column unconstrained logjoint (matching `batched_logjoint_unconstrained`)
and `gradients` is `parameter_count x batch` with `gradients[p, b] = d(logjoint_b)/d(param_p)`
(matching `batched_logjoint_gradient_unconstrained`).

Constructs a fresh `DeviceBatchedWorkspace` each call; use `DeviceBatchedWorkspace` +
`device_batched_logjoint_gradient!` to reuse device buffers. Throws an `ArgumentError`
(pointing at `device_lowering_report`) if the model is not representable on the device;
there is no silent fallback -- the caller may compute the gradient on the CPU instead.
"""
function device_batched_logjoint_gradient(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap();
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=float(eltype(params)),
)
    workspace = DeviceBatchedWorkspace(
        model,
        size(params, 2);
        backend=backend,
        precision=precision,
        args=args,
        constraints=constraints,
    )
    return device_batched_logjoint_gradient!(workspace, params)
end

"""
    device_batched_logjoint_gradient!(workspace, params) -> (values::Vector, gradients::Matrix)

Runs the fused device gradient kernel for `params` (unconstrained, `parameter_count x batch`)
reusing the workspace's staged observations and device buffers (gradient buffers are
allocated on first use), returning `(values, gradients)`. Successive calls with different
`params` are independent (staging is not mutated).
"""
function device_batched_logjoint_gradient!(workspace::DeviceBatchedWorkspace{T}, params::AbstractMatrix) where {T}
    size(params, 1) == workspace.parameter_count || throw(
        _signature_length_error(
            workspace.model,
            workspace.layout,
            workspace.signature_constraints,
            workspace.parameter_count,
            size(params, 1),
        ),
    )
    size(params, 2) == workspace.batch_size ||
        throw(DimensionMismatch("expected $(workspace.batch_size) batch elements, got $(size(params, 2))"))

    _device_upload_params!(workspace, params)
    _device_ensure_gradient_buffers!(workspace)

    # Serial `(P, C)` scan or the observation-parallel tiled path (issue #153),
    # selected by the workspace; one synchronize covers whichever kernels ran.
    _device_launch_gradient!(workspace)
    KernelAbstractions.synchronize(workspace.backend)

    values = Vector{T}(undef, workspace.batch_size)
    gradients = Matrix{T}(undef, workspace.parameter_count, workspace.batch_size)
    copyto!(values, workspace.totals_device)
    copyto!(gradients, workspace.gradients_device)
    return (values, gradients)
end
