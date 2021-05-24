
using NLopt


# Firm value function if they do not change the markup 
function V(para::MParameters,Vprime,μ_,ε)
    @unpack β,γ,W̄,μe,σe = para    
    # Calculating this period's markup given this period's shock and last period's markup
    μ = μ_/exp(ε)
    v = W̄^(1-γ)*(μ - 1)*μ^(-γ) + β*expectation_normal(ε'->Vprime(μ,ε'),μe,σe,10)
    return v, μ
end 


# Firm value function if they do change the markup
function J(para::MParameters)#,Πprime)
    @unpack β,γ,κ,W̄,μe,σe,mugrid = para
    # Value if the firm does change prices  
    J(μo) = -(W̄^(1-γ)*(μo - 1)*μo^(-γ) - κ) #+ β*expectation_normal(ε->Πprime(μo,ε),μe,σe,10))
    Jresult = optimize(J,mugrid[1],mugrid[end])
    return return(μprime = Jresult.minimizer, Π = -Jresult.minimum)
end 

J(para)


""" 
    find_optimal_policy(αvec,Uf,hf)

Computes the optimal policy given policy fuctions hf and indirect utility Uf
"""
function find_optimal_policy(para::MParameters,Πprime, μ_)
    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [0., -Inf])
    upper_bounds!(opt, [20,Inf])
    ftol_rel!(opt,1e-8)

    # Setting up piecewise function 
    min_objective!(opt, x)
    inequality_constraint!(opt, (μ,ε)->V(para, Πprime, μ_, ε) - x)
    inequality_constraint!(opt, (μ,ε)->J(para, Πprime) - x)
    
    
    min_objective!(opt, (x,g)->-government_welfare(x[1],x[2],αvec,Uf))
    equality_constraint!(opt, (x,g) -> -budget_residual(x[1],x[2],αvec,hf))




    minf,minx,ret = NLopt.optimize(opt, [0.3, 0.3])
    if ret == :FTOL_REACHED
        return minx
    end
end;




basis_x = SplineParams(LinRange(-1,1,5),0,3) #cubic splines along x
basis_y = ChebParams(3,-1,1)#Chebyshev polynomials along y

basis = Basis(basis_x,basis_y)


function optimalpolicy(para::MParameters,μ)
    @unpack β,γ,κ,W̄,μe,σe,mugrid = para
    mu_bounds = [mugrid[1],mugrid[end]]
    # Calculating this period's markup given this period's shock and last period's markup
    #μ = μ_/exp(ε)
    # Value if the firm does not change prices  
    V = W̄^(1-γ)*(μ - 1)*μ^(-γ) #+ β*expectation_normal(εprime->Πprime([μ,εprime]),μe,σe,10)
    # Value if the firm does change prices  
    J(μo) = - (W̄^(1-γ)*(μo - 1)*μo^(-γ) - κ) #+ β*expectation_normal(εprime->Πprime([μo,εprime]),μe,σe,10))
    Jresult = optimize(J,mu_bounds[1],mu_bounds[2])
    # Choosing whether to keep prices or change prices 
    if V >= -Jresult.minimum 
        return(μprime = μ, Π = V)
    else 
        return(μprime = Jresult.minimizer, Π = -Jresult.minimum)
    end
end;


plot(
    layer(y->optimalpolicy(para,y).μprime,0,3,color=["FuncDtioDn"]),
    layer(y->y,0,3,color=["yellow"]),
            Guide.xlabel("μ"),Guide.ylabel("Optimal Policy"))




