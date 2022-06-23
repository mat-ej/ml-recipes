import numpy as np
import scipy.optimize as sco
import cvxpy as cvx

b_eps = 1e-8
VERBOSE = False

P_M = np.array(
      [[0.32822376, 0.67177624],
       [0.87922084, 0.12077919],
       [0.70207   , 0.29793   ],
       [0.27129745, 0.72870255],
       [0.71263921, 0.28736079],
       [0.86146849, 0.13853151],
       [0.64640749, 0.35359249],
       [0.67452425, 0.32547575],
       [0.34954542, 0.65045458],
       [0.77150959, 0.22849041],
       [0.77240551, 0.22759449],
       [0.55140245, 0.44859758],
       [0.45131361, 0.54868639],
       [0.74916768, 0.25083229],
       [0.42509836, 0.57490164],
       [0.80688989, 0.19311012],
       [0.66710472, 0.33289525],
       [0.28043318, 0.71956682],
       [0.37209082, 0.62790918],
       [0.55622828, 0.44377169],
       [0.75824255, 0.24175744],
       [0.31265098, 0.68734902],
       [0.57550371, 0.42449629],
       [0.72901785, 0.27098215],
       [0.82910526, 0.17089474],
       [0.60556376, 0.39443624],
       [0.44234663, 0.55765337],
       [0.72555983, 0.2744402 ],
       [0.76714242, 0.23285758],
       [0.69399321, 0.30600682],
       [0.76021719, 0.23978281],
       [0.66816169, 0.33183831],
       [0.73058534, 0.26941466],
       [0.56409132, 0.43590868],
       [0.39592052, 0.60407948],
       [0.61061001, 0.38938996],
       [0.50639755, 0.49360245],
       [0.71422625, 0.28577372],
       [0.611166  , 0.388834  ],
       [0.42736834, 0.57263166],
       [0.6597119 , 0.3402881 ],
       [0.83751237, 0.1624876 ],
       [0.6336149 , 0.3663851 ],
       [0.75390631, 0.24609368],
       [0.8474648 , 0.15253523],
       [0.66335768, 0.33664232],
       [0.63763481, 0.36236519],
       [0.39416307, 0.60583693],
       [0.48570877, 0.51429123],
       [0.67369497, 0.32630503]])

O = np.array(
      [[ 2.18,  1.77],
       [ 1.08, 10.15],
       [ 1.45,  2.75],
       [ 2.85,  1.45],
       [ 1.68,  2.29],
       [ 1.2 ,  4.59],
       [ 1.38,  2.95],
       [ 1.53,  2.5 ],
       [ 2.25,  1.6 ],
       [ 1.2 ,  4.59],
       [ 1.14,  5.  ],
       [ 1.69,  2.29],
       [ 2.88,  1.48],
       [ 1.58,  2.5 ],
       [ 2.54,  1.55],
       [ 1.47,  2.7 ],
       [ 1.65,  2.2 ],
       [ 3.6 ,  1.3 ],
       [ 2.42,  1.64],
       [ 1.2 ,  4.59],
       [ 1.35,  3.2 ],
       [ 2.49,  1.61],
       [ 1.82,  1.98],
       [ 1.4 ,  2.85],
       [ 1.35,  3.2 ],
       [ 1.48,  2.76],
       [ 2.  ,  2.  ],
       [ 1.4 ,  2.95],
       [ 1.55,  2.5 ],
       [ 1.3 ,  3.6 ],
       [ 1.33,  3.7 ],
       [ 1.52,  2.73],
       [ 1.5 ,  2.8 ],
       [ 1.75,  2.  ],
       [ 1.42,  2.9 ],
       [ 1.48,  2.88],
       [ 2.53,  1.59],
       [ 1.5 ,  2.7 ],
       [ 1.8 ,  2.15],
       [ 1.95,  1.95],
       [ 1.55,  2.4 ],
       [ 1.12,  5.5 ],
       [ 1.45,  2.99],
       [ 1.65,  2.35],
       [ 1.37,  3.1 ],
       [ 1.28,  3.75],
       [ 1.5 ,  2.7 ],
       [ 2.1 ,  1.75],
       [ 2.  ,  1.9 ],
       [ 1.65,  2.35]])

def get_mu_var(odds, probs):
    mu = probs * odds - 1
    e1 = probs * (odds - 1) ** 2 + 1 - probs
    e2 = (probs * odds - 1) ** 2
    var = e1 - e2
    return mu, var


def max_sharpe_opps_mask(odds, probs):
    '''
    Select only maximum sharpe ratio bets in each game with a mask.
    :param probs:
    :param odds:
    :return: Max Sharpe mask
    '''
    mu, var = get_mu_var(odds, probs)
    mask = (mu / var) == np.max((mu / var), axis=1)[:, None]
    return mask

def get_mu_sigma_independent(odds, probs):
    '''
    Returns mu and sigma for independent betting opportunities
    :param odds:
    :param probs:
    :return:
    '''
    mu, var = get_mu_var(odds, probs)
    Sigma = np.diag(var)
    return mu, Sigma


def MaxSharpe(odds, probs, riskFreeRate = 0, b_eps = 1e-6):
    '''
    Finds the portfolio of assets providing the maximum Sharpe Ratio
    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    meanReturns, covMatrix = get_mu_sigma_independent(odds, probs)

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(numAssets))

    try:
        opts = sco.minimize(negSharpeRatio, numAssets*[1./numAssets,], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)

        b_star = opts.X
        b_star[b_star < b_eps] = 0
        b_star /= np.sum(b_star)
    except Exception as e:
        print(e)
        b_star = np.zeros(numAssets)

    return b_star

def calcPortfolioPerf(weights, meanReturns, covMatrix):
    '''
    Calculates the expected mean of returns and volatility for a portolio of
    assets, each carrying the weight specified by weights

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    tuple containing the portfolio return and volatility
    '''

    portReturn = np.sum(meanReturns*weights)
    portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))

    return portReturn, portStdDev

def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate=0):

    '''
    Returns the negated Sharpe Ratio for the specified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    p_ret, p_var = calcPortfolioPerf(weights, meanReturns, covMatrix)

    return -(p_ret - riskFreeRate) / p_var

def MaxSharpeConvex(odds, probs, verbose = VERBOSE, b_eps = b_eps):

    n = len(probs)
    b = cvx.Variable(n)
    mu, Sigma = get_mu_sigma_independent(odds, probs)

    # SHARPE modified = minimization of risk only with returns constrained.
    goal = cvx.Minimize(cvx.quad_form(b, Sigma))

    constraints = [
        b.T * mu == 1,
        cvx.sum(b) >= 0,
        b >= 0,
    ]

    problem = cvx.Problem(goal, constraints)
    problem.solve(verbose=verbose)
    b_star = b.value
    b_star /= np.sum(b_star) # first normalization is transformation from convex MaxSharpe to regular
    b_star[b_star < b_eps] = 0
    b_star /= np.sum(b_star) # second norm. is to deal with possible lower bound of betting b_eps
    return b_star

def MaxSharpeBounded(odds, probs, l_bound, rf = 0, b_eps = b_eps, solver ='ECOS_BB', verbose = VERBOSE):
    mu, Sigma = get_mu_sigma_independent(odds, probs)

    n = len(probs)
    '''
    Vars:
        y = portfolio vector 
        K = sum of portfolio vector
        w - boolean switch
        
        b* = y*/K*
    '''

    y = cvx.Variable(n)
    K = cvx.Variable(1)
    w = cvx.Variable(n, boolean=True)

    M = 100 # Big M
    goal = cvx.Minimize(cvx.quad_form(y, Sigma))


    constraints = [
        ((mu - rf).T * y) == 1, # sharpe constraints
        cvx.sum(y) == K,
        y >= 0,
        K >= 0,
        y >= ((l_bound * K) - (M * w)), # bound constraints
        y <= (M * (1 - w))
    ]

    problem = cvx.Problem(goal, constraints)
    problem.solve(solver, verbose = verbose, max_iters = 10000)
    y_star = y.value
    K_star = K.value

    b_star = y_star / K_star
    b_star[b_star < b_eps] = 0
    b_star /= np.sum(b_star)
    return(b_star)

def GetSharpeObjective(odds, probs, b_portfolio):
    '''
    Maximum Sharpe objective function return/risk
    :param odds:
    :param probs:
    :param b_portfolio: chosen portfolio for given odds, probs
    :return: return / risk
    '''
    mu, Sigma = get_mu_sigma_independent(odds, probs)

    p_return = cvx.sum(mu.T * b_portfolio)
    p_risk = cvx.sqrt(cvx.quad_form(b_portfolio, Sigma))

    # -(p_ret - riskFreeRate) / p_var
    objective = p_return / p_risk
    return objective.value


# Lower bound for MaxSharpeBounded
L_BOUND = 0.01
VERBOSE = False

# Selection criteria mask
S = max_sharpe_opps_mask(O, P_M)

# opportunities
probs = P_M[S]
odds = O[S]

# b_eps = 1e-8 -> cut off for values such as 1e-12 in the portfolio
b = MaxSharpe(odds, probs, b_eps=b_eps)
b_convex = MaxSharpeConvex(odds, probs, b_eps=b_eps, verbose = VERBOSE)
b_convex_normalized_bounded = MaxSharpeConvex(odds, probs, b_eps=L_BOUND, verbose = VERBOSE)

b_bounded = MaxSharpeBounded(odds, probs, l_bound=L_BOUND, b_eps=b_eps, verbose = True)





print("\n MaxSharpe SciPy")
print(b)
print("\n MaxSharpeConvex CvxPy")
print(b_convex)

# Euclidean distance between MaxSharpe and MaxSharpeConvex
print("\n dist = %.6f" % (np.linalg.norm(b - b_convex)))

print("\n MaxSharpeConvex with constraint -> investment >= %.3f or investment == 0" % (L_BOUND))
print(b_bounded)
print("\n")
print("SciPy objective = %.15f" % (GetSharpeObjective(odds, probs, b)))
print("CvxPy objective = %.15f" % (GetSharpeObjective(odds, probs, b_convex)))

print("\nCvxPy bounded by normalization objective = %.15f with l_bound = %f" % (GetSharpeObjective(odds, probs, b_convex_normalized_bounded), L_BOUND))
print("MIP objective = %.15f with l_bound = %f" % (GetSharpeObjective(odds, probs, b_bounded), L_BOUND))

print("Improvement of MIP = %.15f" % ((GetSharpeObjective(odds, probs, b_bounded) - GetSharpeObjective(odds, probs, b_convex_normalized_bounded))))




