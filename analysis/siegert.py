import scipy
import scipy.integrate
from matplotlib.pylab import *  # for plot

# firing rate after Brunel & Hakim 1999
# only true for delta shaped PSCs
#

#
# for mu < V_th
#
def siegert1(tau_m, V_th, V_r, mu, sigma):

    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma
   
    def integrand(u):
        if u == 0:
            return exp(-y_th**2)*2*(y_th - y_r)
        else:
            return exp(-(u-y_th)**2) * ( 1.0 - exp(2*(y_r-y_th)*u) ) / u

    lower_bound = y_th
    err_dn = 1.0
    while err_dn > 1e-12 and lower_bound > 1e-16:
        err_dn = integrand(lower_bound)
        if err_dn > 1e-12:            
            lower_bound /= 2

    upper_bound = y_th
    err_up = 1.0
    while err_up > 1e-12:
       err_up = integrand(upper_bound)
       if err_up > 1e-12:
           upper_bound *= 2

    err = max(err_up, err_dn)

    #print 'upper_bound = ', upper_bound
    #print 'lower_bound = ', lower_bound
    #print 'err_dn = ', err_dn
    #print 'err_up = ', err_up

    return 1.0/(exp(y_th**2)*scipy.integrate.quad(integrand, lower_bound, upper_bound)[0] * tau_m)
 
#
# for mu > V_th
#
def siegert2(tau_m, V_th, V_r, mu, sigma):

    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma

    def integrand(u):
        if u == 0:
            return 2*(y_th - y_r)
        else:
            return ( exp(2*y_th*u -u**2) - exp(2*y_r*u -u**2) ) / u

    upper_bound = 1.0
    err = 1.0
    while err > 1e-12:
        err = integrand(upper_bound)
        upper_bound *= 2

    return 1.0/(scipy.integrate.quad(integrand, 0.0, upper_bound)[0] * tau_m)


    
def nu_0(tau_m, V_th, V_r, mu, sigma):
    
    if mu <= V_th*0.95:
        return siegert1(tau_m, V_th, V_r, mu, sigma)
    else:
        return siegert2(tau_m, V_th, V_r, mu, sigma)


def nu_0_tom(tau_m, V_th, V_r, mu, sigma):

    def integrand(u):
        if u < -4.0:
            return -1/sqrt(pi) * ( 1.0/u - 1.0/(2.0*u**3) + 3.0/(4.0*u**5) - 15.0/(8.0*u**7) )
        else:
            return exp(u**2)*(1+scipy.special.erf(u))

    upper_bound = (V_th - mu)/sigma
    lower_bound = (V_r - mu)/sigma

    #dx = 1e-3
    #I = 0.0
    #for x in arange(lower_bound, upper_bound, dx):
    #    I += integrand(x)*dx
    #return 1.0 / (I * tau_m * sqrt(pi))    
    #return 1.0 / ( scipy.integrate.quad(integrand, lower_bound, upper_bound)[0] * tau_m * sqrt(pi) )
    


#
# derivative of nu_0 by mu
#
def d_nu_d_mu(tau_m, V_th, V_r, mu, sigma):

    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma

    nu0 = nu_0(tau_m, V_th, V_r, mu, sigma)
    return sqrt(pi) * tau_m * nu0**2 / sigma * (exp(y_th**2) * (1 + scipy.special.erf(y_th)) - exp(y_r**2) * (1 + scipy.special.erf(y_r)))



def d_nu_d_mu_give_nu0(tau_m, V_th, V_r, mu, sigma, nu0):

    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma

    return sqrt(pi) * tau_m * nu0**2 / sigma * (exp(y_th**2) * (1 + scipy.special.erf(y_th)) - exp(y_r**2) * (1 + scipy.special.erf(y_r)))

#
# membrane potential distribution
#
def pdf_V(x, tau_m, V_r, V_th, nu_0, mu, sigma):
    '''membrane potential distribution'''
    y = (x-mu)/sigma
    y_th = (V_th-mu)/sigma
    y_r = (V_r-mu)/sigma

    def integrand(y):
	return exp(-y**2) * scipy.integrate.quad( lambda u: exp(u**2) , max(y,y_r), y_th )[0]

    return 2.0*nu_0*tau_m/sigma * ( integrand(y) )
