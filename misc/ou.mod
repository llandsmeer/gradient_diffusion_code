NEURON {
    SUFFIX ou
    NONSPECIFIC_CURRENT i
    RANGE mu, theta, sigma
}
PARAMETER {
    mu = 0
    theta = 1
    sigma = 1
}
STATE { x }
INITIAL { x=0 }
WHITE_NOISE { W }
DERIVATIVE dX {
    x' = (mu - x)*theta +  sigma*W
}
BREAKPOINT {
    SOLVE dX METHOD stochastic
    i = - x*x*x*x
}
