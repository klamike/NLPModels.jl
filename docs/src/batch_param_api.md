# Batch Parametric API

This section documents the batch parametric API of `NLPModels.jl`.
It combines the [Batch API](@ref) conventions with the [Parametric API](@ref), providing batched evaluation of derivatives with respect to parameters `p` for problems of the form

```math
\begin{aligned}
\min \quad & f_i(x, p_i) \\
& c_{L,i}(p_i) \leq c_i(x, p_i) \leq c_{U,i}(p_i) \\
& \ell_i(p_i) \leq x \leq u_i(p_i)
\end{aligned}
```

for ``i = 1, \ldots, N_{\text{batch}}``.

Like the standard batch API, all input and output data are organized as matrices where each column corresponds to one problem in the batch.
The sparsity structures (rows, cols) are shared across the batch.

The `BatchNLPModelMeta` struct carries all parametric metadata fields (`nparam`, `nnzjp`, `nnzhp`, etc.) with the same semantics as `NLPModelMeta`.
The availability flags (e.g., `grad_param_available`, `jac_param_available`) default to `false`.

---

## Objective gradient wrt parameters

Evaluate ``\nabla_p f_i(x_i, p_i)`` for each batch element:

| Function | Signature |
|:--------:|:---------:|
| `grad_param`  | `bg = grad_param(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` |
| `grad_param!` | `bg = grad_param!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bg::AbstractMatrix)` |

---

## Sparse constraint Jacobian wrt parameters

Evaluate ``J_{p,i}(x_i) = \nabla_p c_i(x_i, p_i)^T`` in sparse coordinate format:

| Function | Signature |
|:--------:|:---------:|
| `jac_param_structure`  | `(rows, cols) = jac_param_structure(bnlp::AbstractBatchNLPModel)` |
| `jac_param_structure!` | `(rows, cols) = jac_param_structure!(bnlp::AbstractBatchNLPModel, rows::AbstractVector, cols::AbstractVector)` |
| `jac_param_coord`  | `bvals = jac_param_coord(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` |
| `jac_param_coord!` | `bvals = jac_param_coord!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bvals::AbstractMatrix)` |

---

## Constraint Jacobian-vector products wrt parameters

Evaluate products with ``J_{p,i}(x_i)`` and ``J_{p,i}(x_i)^T`` without forming the matrix:

| Function | Signature |
|:--------:|:---------:|
| `jpprod`  | `bJv = jpprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix)` |
| `jpprod!` | `bJv = jpprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix)` |
| `jptprod`  | `bJtv = jptprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix)` |
| `jptprod!` | `bJtv = jptprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix)` |

---

## Sparse variable-parameter Hessian of the Lagrangian

Evaluate ``\nabla^2_{xp} L_i(x_i, y_i, p_i)`` in sparse coordinate format.
The `obj_weight` is provided as a per-batch vector:

| Function | Signature |
|:--------:|:---------:|
| `hess_param_structure`  | `(rows, cols) = hess_param_structure(bnlp::AbstractBatchNLPModel)` |
| `hess_param_structure!` | `(rows, cols) = hess_param_structure!(bnlp::AbstractBatchNLPModel, rows::AbstractVector, cols::AbstractVector)` |
| `hess_param_coord`  | `bvals = hess_param_coord(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector)` |
| `hess_param_coord!` | `bvals = hess_param_coord!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bvals::AbstractMatrix)` |

---

## Variable-parameter Hessian-vector products

Evaluate products with ``\nabla^2_{xp} L_i`` and its transpose:

| Function | Signature |
|:--------:|:---------:|
| `hpprod`  | `bHv = hpprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector)` |
| `hpprod!` | `bHv = hpprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix)` |
| `hptprod`  | `bHtv = hptprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector)` |
| `hptprod!` | `bHtv = hptprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHtv::AbstractMatrix)` |

---

## Sparse constraint lower-bound Jacobian wrt parameters

Evaluate ``\nabla_p c_{L,i}(p_i)`` for each batch element:

| Function | Signature |
|:--------:|:---------:|
| `lcon_jac_param_structure`  | `(rows, cols) = lcon_jac_param_structure(bnlp::AbstractBatchNLPModel)` |
| `lcon_jac_param_structure!` | `(rows, cols) = lcon_jac_param_structure!(bnlp::AbstractBatchNLPModel, rows::AbstractVector, cols::AbstractVector)` |
| `lcon_jac_param_coord`  | `bvals = lcon_jac_param_coord(bnlp::AbstractBatchNLPModel)` |
| `lcon_jac_param_coord!` | `bvals = lcon_jac_param_coord!(bnlp::AbstractBatchNLPModel, bvals::AbstractMatrix)` |
| `lcon_jpprod`  | `bJv = lcon_jpprod(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix)` |
| `lcon_jpprod!` | `bJv = lcon_jpprod!(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix, bJv::AbstractMatrix)` |
| `lcon_jptprod`  | `bJtv = lcon_jptprod(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix)` |
| `lcon_jptprod!` | `bJtv = lcon_jptprod!(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix, bJtv::AbstractMatrix)` |

---

## Sparse constraint upper-bound Jacobian wrt parameters

Evaluate ``\nabla_p c_{U,i}(p_i)`` for each batch element:

| Function | Signature |
|:--------:|:---------:|
| `ucon_jac_param_structure`  | `(rows, cols) = ucon_jac_param_structure(bnlp::AbstractBatchNLPModel)` |
| `ucon_jac_param_structure!` | `(rows, cols) = ucon_jac_param_structure!(bnlp::AbstractBatchNLPModel, rows::AbstractVector, cols::AbstractVector)` |
| `ucon_jac_param_coord`  | `bvals = ucon_jac_param_coord(bnlp::AbstractBatchNLPModel)` |
| `ucon_jac_param_coord!` | `bvals = ucon_jac_param_coord!(bnlp::AbstractBatchNLPModel, bvals::AbstractMatrix)` |
| `ucon_jpprod`  | `bJv = ucon_jpprod(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix)` |
| `ucon_jpprod!` | `bJv = ucon_jpprod!(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix, bJv::AbstractMatrix)` |
| `ucon_jptprod`  | `bJtv = ucon_jptprod(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix)` |
| `ucon_jptprod!` | `bJtv = ucon_jptprod!(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix, bJtv::AbstractMatrix)` |

---

## Sparse variable lower-bound Jacobian wrt parameters

Evaluate ``\nabla_p \ell_i(p_i)`` for each batch element:

| Function | Signature |
|:--------:|:---------:|
| `lvar_jac_param_structure`  | `(rows, cols) = lvar_jac_param_structure(bnlp::AbstractBatchNLPModel)` |
| `lvar_jac_param_structure!` | `(rows, cols) = lvar_jac_param_structure!(bnlp::AbstractBatchNLPModel, rows::AbstractVector, cols::AbstractVector)` |
| `lvar_jac_param_coord`  | `bvals = lvar_jac_param_coord(bnlp::AbstractBatchNLPModel)` |
| `lvar_jac_param_coord!` | `bvals = lvar_jac_param_coord!(bnlp::AbstractBatchNLPModel, bvals::AbstractMatrix)` |
| `lvar_jpprod`  | `bJv = lvar_jpprod(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix)` |
| `lvar_jpprod!` | `bJv = lvar_jpprod!(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix, bJv::AbstractMatrix)` |
| `lvar_jptprod`  | `bJtv = lvar_jptprod(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix)` |
| `lvar_jptprod!` | `bJtv = lvar_jptprod!(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix, bJtv::AbstractMatrix)` |

---

## Sparse variable upper-bound Jacobian wrt parameters

Evaluate ``\nabla_p u_i(p_i)`` for each batch element:

| Function | Signature |
|:--------:|:---------:|
| `uvar_jac_param_structure`  | `(rows, cols) = uvar_jac_param_structure(bnlp::AbstractBatchNLPModel)` |
| `uvar_jac_param_structure!` | `(rows, cols) = uvar_jac_param_structure!(bnlp::AbstractBatchNLPModel, rows::AbstractVector, cols::AbstractVector)` |
| `uvar_jac_param_coord`  | `bvals = uvar_jac_param_coord(bnlp::AbstractBatchNLPModel)` |
| `uvar_jac_param_coord!` | `bvals = uvar_jac_param_coord!(bnlp::AbstractBatchNLPModel, bvals::AbstractMatrix)` |
| `uvar_jpprod`  | `bJv = uvar_jpprod(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix)` |
| `uvar_jpprod!` | `bJv = uvar_jpprod!(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix, bJv::AbstractMatrix)` |
| `uvar_jptprod`  | `bJtv = uvar_jptprod(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix)` |
| `uvar_jptprod!` | `bJtv = uvar_jptprod!(bnlp::AbstractBatchNLPModel, bv::AbstractMatrix, bJtv::AbstractMatrix)` |
