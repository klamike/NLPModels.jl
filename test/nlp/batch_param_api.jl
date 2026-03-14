@testset "Batch Parametric API" begin
  params = [2.0 3.0 4.0; 3.0 4.0 5.0]
  bnlp = BatchSimpleParamNLPModel(params)
  models = [SimpleParamNLPModel(; ps = params[:, i]) for i = 1:3]
  nbatch = 3

  @test bnlp.meta.nbatch == nbatch
  @test bnlp.meta.nparam == 2

  bx = [1.0 3.0 5.0; 2.0 4.0 6.0]
  by = [-1.0 -3.0 -5.0;]
  bv_p = [0.3 0.5 0.7; 0.7 0.9 1.1]
  bv_x = [0.4 0.6 0.8; 0.6 0.8 1.0]
  bv_c = reshape([-1.2, -2.4, -3.6], 1, 3)
  bobj_weight = [1.0, 2.0, 3.0]

  @testset "grad_param" begin
    bg = grad_param(bnlp, bx)
    for i = 1:nbatch
      @test bg[:, i] ≈ grad_param(models[i], bx[:, i])
    end
  end

  @testset "jac_param_structure" begin
    rows, cols = jac_param_structure(bnlp)
    rows_i, cols_i = jac_param_structure(models[1])
    @test rows == rows_i
    @test cols == cols_i
  end

  @testset "jac_param_coord" begin
    bvals = jac_param_coord(bnlp, bx)
    for i = 1:nbatch
      @test bvals[:, i] ≈ jac_param_coord(models[i], bx[:, i])
    end
  end

  @testset "jpprod" begin
    bJv = jpprod(bnlp, bx, bv_p)
    for i = 1:nbatch
      @test bJv[:, i] ≈ jpprod(models[i], bx[:, i], bv_p[:, i])
    end
  end

  @testset "jptprod" begin
    bJtv = jptprod(bnlp, bx, bv_c)
    for i = 1:nbatch
      @test bJtv[:, i] ≈ jptprod(models[i], bx[:, i], bv_c[:, i])
    end
  end

  @testset "hess_param_structure" begin
    rows, cols = hess_param_structure(bnlp)
    rows_i, cols_i = hess_param_structure(models[1])
    @test rows == rows_i
    @test cols == cols_i
  end

  @testset "hess_param_coord" begin
    bvals = hess_param_coord(bnlp, bx, by, bobj_weight)
    for i = 1:nbatch
      @test bvals[:, i] ≈ hess_param_coord(models[i], bx[:, i], by[:, i]; obj_weight = bobj_weight[i])
    end
  end

  @testset "hpprod" begin
    bHv = hpprod(bnlp, bx, by, bv_p, bobj_weight)
    for i = 1:nbatch
      @test bHv[:, i] ≈ hpprod(models[i], bx[:, i], by[:, i], bv_p[:, i]; obj_weight = bobj_weight[i])
    end
  end

  @testset "hptprod" begin
    bHtv = hptprod(bnlp, bx, by, bv_x, bobj_weight)
    for i = 1:nbatch
      @test bHtv[:, i] ≈ hptprod(models[i], bx[:, i], by[:, i], bv_x[:, i]; obj_weight = bobj_weight[i])
    end
  end

  @testset "lcon_jac" begin
    rows, cols = lcon_jac_param_structure(bnlp)
    rows_i, cols_i = lcon_jac_param_structure(models[1])
    @test rows == rows_i
    @test cols == cols_i

    bvals = lcon_jac_param_coord(bnlp)
    for i = 1:nbatch
      @test bvals[:, i] ≈ lcon_jac_param_coord(models[i])
    end

    bJv = lcon_jpprod(bnlp, bv_p)
    for i = 1:nbatch
      @test bJv[:, i] ≈ lcon_jpprod(models[i], bv_p[:, i])
    end

    bJtv = lcon_jptprod(bnlp, bv_c)
    for i = 1:nbatch
      @test bJtv[:, i] ≈ lcon_jptprod(models[i], bv_c[:, i])
    end
  end

  @testset "ucon_jac" begin
    rows, cols = ucon_jac_param_structure(bnlp)
    rows_i, cols_i = ucon_jac_param_structure(models[1])
    @test rows == rows_i
    @test cols == cols_i

    bvals = ucon_jac_param_coord(bnlp)
    for i = 1:nbatch
      @test bvals[:, i] ≈ ucon_jac_param_coord(models[i])
    end

    bJv = ucon_jpprod(bnlp, bv_p)
    for i = 1:nbatch
      @test bJv[:, i] ≈ ucon_jpprod(models[i], bv_p[:, i])
    end

    bJtv = ucon_jptprod(bnlp, bv_c)
    for i = 1:nbatch
      @test bJtv[:, i] ≈ ucon_jptprod(models[i], bv_c[:, i])
    end
  end

  @testset "lvar_jac" begin
    rows, cols = lvar_jac_param_structure(bnlp)
    rows_i, cols_i = lvar_jac_param_structure(models[1])
    @test rows == rows_i
    @test cols == cols_i

    bvals = lvar_jac_param_coord(bnlp)
    for i = 1:nbatch
      @test bvals[:, i] ≈ lvar_jac_param_coord(models[i])
    end

    bJv = lvar_jpprod(bnlp, bv_p)
    for i = 1:nbatch
      @test bJv[:, i] ≈ lvar_jpprod(models[i], bv_p[:, i])
    end

    bJtv = lvar_jptprod(bnlp, bv_x)
    for i = 1:nbatch
      @test bJtv[:, i] ≈ lvar_jptprod(models[i], bv_x[:, i])
    end
  end

  @testset "uvar_jac" begin
    rows, cols = uvar_jac_param_structure(bnlp)
    rows_i, cols_i = uvar_jac_param_structure(models[1])
    @test rows == rows_i
    @test cols == cols_i

    bvals = uvar_jac_param_coord(bnlp)
    for i = 1:nbatch
      @test bvals[:, i] ≈ uvar_jac_param_coord(models[i])
    end

    bJv = uvar_jpprod(bnlp, bv_p)
    for i = 1:nbatch
      @test bJv[:, i] ≈ uvar_jpprod(models[i], bv_p[:, i])
    end

    bJtv = uvar_jptprod(bnlp, bv_x)
    for i = 1:nbatch
      @test bJtv[:, i] ≈ uvar_jptprod(models[i], bv_x[:, i])
    end
  end
end
