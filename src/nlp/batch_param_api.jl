function grad_param(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bg = S(undef, bnlp.meta.nparam, bnlp.meta.nbatch)
  grad_param!(bnlp, bx, bg)
  return bg
end

function jac_param_structure(bnlp::AbstractBatchNLPModel)
  rows = Vector{Int}(undef, bnlp.meta.nnzjp)
  cols = Vector{Int}(undef, bnlp.meta.nnzjp)
  jac_param_structure!(bnlp, rows, cols)
  return (rows, cols)
end

function jac_param_coord(bnlp::AbstractBatchNLPModel{T, S}, bx::AbstractMatrix) where {T, S}
  bvals = S(undef, bnlp.meta.nnzjp, bnlp.meta.nbatch)
  jac_param_coord!(bnlp, bx, bvals)
  return bvals
end

function jpprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractMatrix,
  bv::AbstractMatrix,
) where {T, S}
  bJv = S(undef, bnlp.meta.ncon, bnlp.meta.nbatch)
  jpprod!(bnlp, bx, bv, bJv)
  return bJv
end

function jptprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractMatrix,
  bv::AbstractMatrix,
) where {T, S}
  bJtv = S(undef, bnlp.meta.nparam, bnlp.meta.nbatch)
  jptprod!(bnlp, bx, bv, bJtv)
  return bJtv
end

function hess_param_structure(bnlp::AbstractBatchNLPModel)
  rows = Vector{Int}(undef, bnlp.meta.nnzhp)
  cols = Vector{Int}(undef, bnlp.meta.nnzhp)
  hess_param_structure!(bnlp, rows, cols)
  return (rows, cols)
end

function hess_param_coord(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
) where {T, S}
  bvals = S(undef, bnlp.meta.nnzhp, bnlp.meta.nbatch)
  hess_param_coord!(bnlp, bx, by, bobj_weight, bvals)
  return bvals
end

function hpprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bv::AbstractMatrix,
  bobj_weight::AbstractVector,
) where {T, S}
  bHv = S(undef, bnlp.meta.nvar, bnlp.meta.nbatch)
  hpprod!(bnlp, bx, by, bv, bobj_weight, bHv)
  return bHv
end

function hptprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bv::AbstractMatrix,
  bobj_weight::AbstractVector,
) where {T, S}
  bHtv = S(undef, bnlp.meta.nparam, bnlp.meta.nbatch)
  hptprod!(bnlp, bx, by, bv, bobj_weight, bHtv)
  return bHtv
end

# Bound Jacobians wrt parameters

function lcon_jac_param_structure(bnlp::AbstractBatchNLPModel)
  rows = Vector{Int}(undef, bnlp.meta.nnzjplcon)
  cols = Vector{Int}(undef, bnlp.meta.nnzjplcon)
  lcon_jac_param_structure!(bnlp, rows, cols)
  return (rows, cols)
end

function lcon_jac_param_coord(bnlp::AbstractBatchNLPModel{T, S}) where {T, S}
  bvals = S(undef, bnlp.meta.nnzjplcon, bnlp.meta.nbatch)
  lcon_jac_param_coord!(bnlp, bvals)
  return bvals
end

function lcon_jpprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bv::AbstractMatrix,
) where {T, S}
  bJv = S(undef, bnlp.meta.ncon, bnlp.meta.nbatch)
  lcon_jpprod!(bnlp, bv, bJv)
  return bJv
end

function lcon_jptprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bv::AbstractMatrix,
) where {T, S}
  bJtv = S(undef, bnlp.meta.nparam, bnlp.meta.nbatch)
  lcon_jptprod!(bnlp, bv, bJtv)
  return bJtv
end

function ucon_jac_param_structure(bnlp::AbstractBatchNLPModel)
  rows = Vector{Int}(undef, bnlp.meta.nnzjpucon)
  cols = Vector{Int}(undef, bnlp.meta.nnzjpucon)
  ucon_jac_param_structure!(bnlp, rows, cols)
  return (rows, cols)
end

function ucon_jac_param_coord(bnlp::AbstractBatchNLPModel{T, S}) where {T, S}
  bvals = S(undef, bnlp.meta.nnzjpucon, bnlp.meta.nbatch)
  ucon_jac_param_coord!(bnlp, bvals)
  return bvals
end

function ucon_jpprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bv::AbstractMatrix,
) where {T, S}
  bJv = S(undef, bnlp.meta.ncon, bnlp.meta.nbatch)
  ucon_jpprod!(bnlp, bv, bJv)
  return bJv
end

function ucon_jptprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bv::AbstractMatrix,
) where {T, S}
  bJtv = S(undef, bnlp.meta.nparam, bnlp.meta.nbatch)
  ucon_jptprod!(bnlp, bv, bJtv)
  return bJtv
end

function lvar_jac_param_structure(bnlp::AbstractBatchNLPModel)
  rows = Vector{Int}(undef, bnlp.meta.nnzjplvar)
  cols = Vector{Int}(undef, bnlp.meta.nnzjplvar)
  lvar_jac_param_structure!(bnlp, rows, cols)
  return (rows, cols)
end

function lvar_jac_param_coord(bnlp::AbstractBatchNLPModel{T, S}) where {T, S}
  bvals = S(undef, bnlp.meta.nnzjplvar, bnlp.meta.nbatch)
  lvar_jac_param_coord!(bnlp, bvals)
  return bvals
end

function lvar_jpprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bv::AbstractMatrix,
) where {T, S}
  bJv = S(undef, bnlp.meta.nvar, bnlp.meta.nbatch)
  lvar_jpprod!(bnlp, bv, bJv)
  return bJv
end

function lvar_jptprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bv::AbstractMatrix,
) where {T, S}
  bJtv = S(undef, bnlp.meta.nparam, bnlp.meta.nbatch)
  lvar_jptprod!(bnlp, bv, bJtv)
  return bJtv
end

function uvar_jac_param_structure(bnlp::AbstractBatchNLPModel)
  rows = Vector{Int}(undef, bnlp.meta.nnzjpuvar)
  cols = Vector{Int}(undef, bnlp.meta.nnzjpuvar)
  uvar_jac_param_structure!(bnlp, rows, cols)
  return (rows, cols)
end

function uvar_jac_param_coord(bnlp::AbstractBatchNLPModel{T, S}) where {T, S}
  bvals = S(undef, bnlp.meta.nnzjpuvar, bnlp.meta.nbatch)
  uvar_jac_param_coord!(bnlp, bvals)
  return bvals
end

function uvar_jpprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bv::AbstractMatrix,
) where {T, S}
  bJv = S(undef, bnlp.meta.nvar, bnlp.meta.nbatch)
  uvar_jpprod!(bnlp, bv, bJv)
  return bJv
end

function uvar_jptprod(
  bnlp::AbstractBatchNLPModel{T, S},
  bv::AbstractMatrix,
) where {T, S}
  bJtv = S(undef, bnlp.meta.nparam, bnlp.meta.nbatch)
  uvar_jptprod!(bnlp, bv, bJtv)
  return bJtv
end
