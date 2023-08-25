#############################           Alessandro Canzonieri               ##########################
# The code is a bit rough and not well commented
# I have not been able to reproduce figure 4 and it hink there is an error on the calculations




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import scipy as py
from scipy.optimize import curve_fit
import os
import pandas as pd
import arch
from scipy.special import gamma, factorial


INFORMED_TRADERS = 0.035 # Fraction of Informed Traders in one experiment equivalent to the number of non random transaction on the amrket (example)

sqrt2= np.sqrt(2)
def p_t(chi):
    return  0.5 + 0.5*py.special.erf(chi/sqrt2)      #sarebbe l'equazione (11) del paper, dove chi è una variabile random gaussianamente distribuita con media nu*sqrt(t)
                                                         #e varianza var=(1-nu**2)
'''
def p_t_k(chi, k):      #la seconda approssimazione. Ho verificato che non funziona
    return (0.5 + (k*(3.1415926/4)*py.special.erfi(chi/sqrt2)))
'''



def p_t_k(chi, k):      #la terza approssimazione
    return (0.5 + k * ((np.sqrt(3.14)/(2*sqrt2)) * (chi + ((chi**3)/(6))  +  ((chi**5)/40))  ))



def p_t_k_1(chi, k):      #la più completa, o prima approssimazione
    return (0.5+((py.special.gamma(0.5*(1+k)))/(sqrt2*py.special.gamma(0.5*k)))*(chi+((1-2*k)/6)*(chi**3)+(((3-4*k)*(1-4*k))/(120))*(chi**5)) )



def var(chi):
    return (p_t(chi**2)-((p_t(chi))**2))


def p_t_bar(chi, nu_bar_t):
    return ((py.special.erf(chi/sqrt2) + py.special.erf((nu_bar_t -chi)/sqrt2))/((py.special.erf((nu_bar_t + chi)/sqrt2)) + py.special.erf((nu_bar_t - chi)/sqrt2)))


                            ###############                             Decay                       #################

def p_t_decay_buy(t, Q , nu):              #price decay till I buy
    return (0.5 + py.special.erf(Q/(np.sqrt(4*t-2*nu*Q))))

def p_t_decay_simplified(t, Q , nu):              #price decay till I sell
    return (0.5 + 0.5 * py.special.erf(Q/(2* np.sqrt(t))))

                            ###############                         end Decay                   #######################

def decay(x, Q,c):                    #funzione di decadimento
    return (Q/(2*np.sqrt(3.14*x)))+c




price_sim=[]
var_prices=[]
random_chi =[]
delta_prices=[]
std_t = []
time_q=[]
xx=[]
sigma=[]
t_lim_array=[]


for q in range (1,80,1):                    #montecarlo simulation for nu fixed and Q variable
    t_lim = int(q/INFORMED_TRADERS)
    t_lim_array.append(t_lim)
    xx.append(q)
    sigma.append((0.001))
    chi0 = ((np.random.normal(INFORMED_TRADERS*np.sqrt(1),np.sqrt(1-INFORMED_TRADERS**2),100000)))
    chi_t_lim = ((np.random.normal(INFORMED_TRADERS*np.sqrt(t_lim), np.sqrt(1-INFORMED_TRADERS**2),100000)))

    p_t_avg=(((p_t(chi_t_lim) ).sum())/100000)
    delta_p_t= (p_t(chi_t_lim))

    delta_prices.append(((p_t(chi_t_lim) - p_t(chi0)).sum())/100000)
    std_t.append( np.sqrt(      (((delta_p_t - p_t_avg)**2).sum())/100000) )








time_nu=[]
std_t_Q=[]
delta_prices_Q=[]
Q = 10
x = np.logspace(-2.3, 0, 25)
for i in range (0, 25,1):                       #montecarlo simulation for Q fixed and nu variable
    nu=x[i]
    t_max = Q/nu
    chi0_Q= ((np.random.normal(x[0]*np.sqrt(1), np.sqrt(1-x[0]**2), 100000)))
    chi_t_lim_Q= ((np.random.normal(nu*np.sqrt(t_max), np.sqrt(1-nu**2), 100000)))

    delta_p_Q_t = p_t(chi_t_lim_Q)
    p_t_Q_avg = ((p_t(chi_t_lim_Q)).sum())/100000       #non plotta la std di p_t - p_0 m aplotta la std di p_t e basta. C'è un errore nel paper


    p_t_Q_avg_2 = ((p_t(chi_t_lim_Q) - p_t(chi0_Q)).sum())/100000
    delta_prices_Q.append(p_t_Q_avg_2)

    std_t_Q.append (np.sqrt(  ((((delta_p_Q_t - p_t_Q_avg)**2).sum())/100000    )))




############################################                simulation  page 9                        ########################################

nu_bar = 0.05
nu= 0.01
Expected_p_t_bar=[]
for t in range (1, 10000, 1):
    chi0_bar= (np.random.normal(nu*np.sqrt(1),np.sqrt(1-nu**2),50000))
    chi_t_bar= (np.random.normal(nu*np.sqrt(t),np.sqrt(1-nu**2),50000))
    nu_bar_t = (nu_bar * np.sqrt(t))
    Expected_p_t_bar.append( ((p_t_bar(chi_t_bar, nu_bar_t) - p_t_bar(chi0_bar, nu_bar)).sum())/50000 )







############################################                fine seconda simulazione            ##############################

###############################################                 simulation impact decay            ##########################

t_buy= 400

Expected_p_t_decay =[]
delta_p_t_decay = []
for t in range (1, 2000, 1):
    Q = INFORMED_TRADERS* t_buy
    if t <= t_buy:                  #buy-time
        q= t * INFORMED_TRADERS
        chi0_decay_buy = ((np.random.normal(nu*np.sqrt(1), np.sqrt(1-nu**2), 1000000)))
        chi_t_decay_buy = ((np.random.normal((q/np.sqrt(t)), np.sqrt(1-((nu*q)/t)), 1000000)))
        delta_p_t_decay.append( (((p_t(chi_t_decay_buy)).sum()) - (p_t(chi0_decay_buy).sum()))/1000000 )


    else:
        chi0_decay_buy = ((np.random.normal(nu*np.sqrt(1), np.sqrt(1-nu**2), 1000000)))
        chi_t_decay_buy = ((np.random.normal((Q/np.sqrt(t)), np.sqrt(1-((nu*Q)/t)), 1000000)))
        delta_p_t_decay.append( (((p_t(chi_t_decay_buy)).sum()) - (p_t(chi0_decay_buy).sum()))/1000000 )


Q = 10



##############################################                  end impact decay                    #######################

##################################################              starting fitting                      #####################################

def sqrtlaw1(x, B):
    return (B*np.sqrt(x))



def polinomlaw(x, B, d):
    return (B*(x**d))


def linefit(x, m, q):
    return (m*x +q)


par1=(1)                ########## fitto la differenza dei prezzi del primo grafico con una legge quadratica
xxa=np.linspace(1,len(delta_prices[0:int(1/INFORMED_TRADERS)]),len(delta_prices[0:int(1/INFORMED_TRADERS)]))
popt1, pcov1 = curve_fit(sqrtlaw1, xxa, delta_prices[0:int(1/INFORMED_TRADERS)],   p0=par1 )
B0= popt1
dB = np.sqrt(pcov1.diagonal())
chisq_sqrtlaw1 =(((delta_prices[0:int(1/INFORMED_TRADERS)] - sqrtlaw1(xxa, B0))**2)).sum()


#################                           fitto la differenza di prezzi del secondo grafico
par2=(1)
#xx=np.linspace(,90,len(delta_prices_Q))
popt2, pcov2 = curve_fit(sqrtlaw1, x[0:19], delta_prices_Q[0:19],  par2, absolute_sigma=False)
B20= popt2
dB2 = np.sqrt(pcov2.diagonal())
chisq_sqrtlaw2 = (((delta_prices_Q[0:19] - sqrtlaw1(x[0:19], B20))**2)).sum()



                                            ###### fitting di nu_bar
t_trans=1/(nu_bar**2)
Expected_p_t_bar_A = Expected_p_t_bar[0:int(t_trans)]
Expected_p_t_bar_B = Expected_p_t_bar[int(t_trans):]
tt_A= np.linspace(2,len(Expected_p_t_bar_A),len(Expected_p_t_bar_A) )
tt_B= np.linspace(2,len(Expected_p_t_bar_B),len(Expected_p_t_bar_B) )

par3 = (1,1)
popt3 , pcov3 = curve_fit(polinomlaw , tt_A, Expected_p_t_bar_A , p0 = par3 )
B30, d30 = popt3
dB3, dd3 = np.sqrt(pcov3.diagonal())
chisq_nu_bar_A = ((Expected_p_t_bar_A - polinomlaw(tt_A, B30, d30))**2).sum()

par4 = (1,1)                #fitto la seconda parte dei dati
popt4 , pcov4 = curve_fit(polinomlaw , tt_B, Expected_p_t_bar_B , p0 = par4 )
B40, d40 = popt4
dB4, dd4 = np.sqrt(pcov4.diagonal())
chisq_nu_bar_B = ((Expected_p_t_bar_B - polinomlaw(tt_B, B40, d40))**2).sum()


                                    ################                        fitting decay

par6=(1,1)
tt_c = np.linspace(0, 399, 400)
popt6 , pcov6 = curve_fit(polinomlaw , tt_c, delta_p_t_decay[0:400] , p0 = par6 )
B60, d60 = popt6
dB6, dd6 = np.sqrt(pcov6.diagonal())
chisq_decay_1 = ((delta_p_t_decay[0:400] - polinomlaw(tt_c, B60, d60))**2).sum()



par7=(1,1)
tt_d = np.linspace( 399,2000, 1600)
popt7 , pcov7 = curve_fit(polinomlaw , tt_d, delta_p_t_decay[399:2000] , p0 = par7 )
B70, d70 = popt7
dB7, dd7 = np.sqrt(pcov7.diagonal())
chisq_decay_2 = ((delta_p_t_decay[399:2000] - polinomlaw(tt_d, B70, d70))**2).sum()






##################################################              fine fitting                      #####################################

########################################                    ploting                                 ######################################
plt.figure(1)
plt.xscale('log')                             #plotto il grafico di p_t con nu fissato e Q variabile
plt.yscale('log')
plt.title("nu = 0.035")
plt.ylabel("E[P_t -p_1] ")
plt.xlabel("Q")
plt.ylim(0.04, 1.1)
plt.xlim(0.9, 91)
plt.errorbar(np.linspace(1,len(delta_prices),len(delta_prices)), delta_prices,  color="blue", fmt=".")     #plotto dati simulati con distribuzioni
plt.errorbar(np.linspace(1,len(std_t),len(std_t)), std_t,  color="red", fmt=".")
tt=np.linspace(0,90,100000)
plt.plot(tt, sqrtlaw1(tt, B0), color = "black", linestyle = '--')     #plotto il fit della differenza dei prezzi con una legge quadratica
plt.axvline(1/INFORMED_TRADERS, color="red", linestyle = '--')




plt.figure(2)                       #plotto il grafico con nu variabile e Q=10 fissato
plt.title("Q = 10")
plt.ylabel("E[P_t -p_1] ")
plt.xlabel("nu")
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.01, 1)
plt.xlim(0.001, 1.1)
t=np.linspace(0.0001,1.1,len(delta_prices_Q))
plt.errorbar(x, delta_prices_Q,  color="blue", fmt=".")     #plotto dati simulati con distribuzioni
plt.errorbar(x, std_t_Q,  color="red", fmt=".")
tt=np.linspace(0.0001,1.1,10000)
plt.plot(tt, sqrtlaw1(tt, B20 ), color = "black", linestyle = '--')           #plotto il fit della differenza dei prezzi
#plt.plot(tt, linefit(tt, m10, q10 ), color = "salmon")      #plotto il fit della varianza dei prezzi di Q=10
plt.axvline(1/(Q), color="green", linestyle = '--')








plt.figure(3)                   #plotto nu_bar
plt.title("nu_bar = 0.05")
plt.ylabel("E[P_t -p_1] ")
plt.xlabel("t")
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.00009, 1)
plt.xlim(0.9, 10500)
tt=np.linspace(1,9999,9999)
print(len(tt))
plt.errorbar(tt, Expected_p_t_bar,  color="blue", fmt="-")
plt.plot((tt), (polinomlaw(tt, B30, d30)),color="black", linestyle = '--')                #plotto il fit della prima parte dei dati
plt.plot((tt), (polinomlaw(tt, B40, d40)),color="salmon" , linestyle = '--')     #plotto il fit della seconda parte dei dati

plt.axvline(1/(nu_bar**2), color="red", linestyle = '--')
plt.axvline(1/(0.01**2), color="red", linestyle = '--')




plt.figure(4)               #plotto il decay dei prezzi
plt.title("decay")
plt.ylabel("E[P_t -p_1] ")
plt.xlabel("t")
plt.ylim(0, 0.25)
plt.xlim(0, 2000)
tt=np.linspace(1,1999,1999)
plt.errorbar(tt, delta_p_t_decay,  color="blue", fmt="-")
plt.plot((tt), (polinomlaw(tt, B60, d60)),color="black" , linestyle = 'dashed')
plt.plot((tt), (polinomlaw(tt, B70, d70)),color="red" , linestyle = 'dashed')


'''
plt.figure(5)
plt.xscale('log')                             #plotto il grafico al variare dei k
plt.yscale('log')
plt.title("price impact for different k  [nu = 0.1]")
plt.ylabel("E[P_t -p_1] al variare di k")
plt.xlabel("t")
plt.ylim(0.0001, 100)
plt.xlim(0.8, 7000)
plt.errorbar(ttt, delta_p_t_k_0,  color="blue", fmt="-")     #plotto dati simulati con distribuzioni al variare di k
plt.errorbar(ttt, delta_p_t_k_1,  color="red", fmt="-")
plt.errorbar(ttt, delta_p_t_k_2,  color="salmon", fmt="-")
plt.errorbar(ttt, delta_p_t_k_3,  color="gray", fmt="-")
#plt.errorbar(ttt, delta_p_t_k_4,  color="black", fmt="-")
plt.errorbar(ttt, delta_p_t_k_5,  color="orange", fmt="-")
'''


#####################                                               printo la errorbar                  ##############################

print("primo fit prima figura", "\n" ,"B +/- db= ", B0,"+/-", dB, "\n", "chisq_1 =",chisq_sqrtlaw1 ,"\n")
print("secondo fit seconda figura", "\n" ,"B +/- db= ", B20,"+/-", dB2 , "\n", "chisq_2 =", chisq_sqrtlaw2 ,"\n\n")

print("paramentri di nu_bar, prima parte", "\n")
print("B30 +/- d30 = ", B30, "+/-", dB3, "\n","d30 +/- dd3 = ",d30, "+/-", dd3,"\n", "chisq_nu_bar_A = ", chisq_nu_bar_A ,"\n" )
print("paramentri di nu_bar, seconda parte", "\n")
print("B30 +/- d30 = ", B40, "+/-", dB4, "\n", "d40 +/- dd4 = ",d40, "+/-", dd4,"\n" "chisq_nu_bar_B = ", chisq_nu_bar_B ,"\n\n" )

print("paramentri di decay", "\n")
print("paramentri di decay, upward", "\n")
print("B60 +/- d60 = ", B60, "+/-", dB6, "\n","d60 +/- dd6 = ",d60, "+/-", dd6,"\n", "chisq_decay_1 = ", chisq_decay_1 ,"\n" )
print("paramentri decay, downward", "\n")
print("B70 +/- d70 = ", B70, "+/-", dB7, "\n", "d70 +/- dd7 = ",d70, "+/-", dd7,"\n" "chisq_decay_2 = ", chisq_decay_2 ,"\n\n" )





plt.show()


############                                        k-impact                                                              ####################
#  I have not been able to reproduce this graph

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import scipy as py
from scipy.optimize import curve_fit
import os
import pandas as pd
import arch
from scipy.special import gamma, factorial



sqrt2= np.sqrt(2)


def cumfunction(a, b , mean , std):
    return py.stats.norm.cdf(b, mean, std)-py.stats.norm.cdf(a, mean, std)


def p_t_k_2(chi, k):
    return (0.5 + (chi/sqrt2) * ((py.special.gamma((1+k)/2))/(py.special.gamma(k/2))) * ((cumfunction((1-(k/2)), 3/2, 0, 1))/(cumfunction(((1-k)/2), 1/2, 0, 1))))


'''
def p_t_k_1(chi, k):      #la seconda approssimazione. Ho verificato che non funziona
    return (0.5 + ((k*(np.pi/4))*py.special.erfi(chi/sqrt2)))
'''


'''
def Expct_Chi(t,n):
    return (((2**(n+1))/np.sqrt(np.pi))*py.special.gamma((n+3)/2)* nu*np.sqrt(t))
'''

def p_t_k(chi, k):      #la terza approssimazione
    return (0.5 + k * ((np.sqrt(np.pi)/(2*sqrt2)) * (chi + ((chi**3)/(6))  +  ((chi**5)/40))  ))



def p_t_k_1(chi, k):      #la più completa, o prima approssimazione
    return (0.5 + (((py.special.gamma((1+k)/2))/(sqrt2*py.special.gamma(k/2)))*( (chi) + (((1-2*k)/6)*(chi**3)) + ((((3-4*k)*(1-4*k))/(120))*(chi**5)) ) ))

'''
def p_t_k_1_exp(chi, k, t):
    return (0.5 + (((py.special.gamma(0.5*(1+k)))/(sqrt2*py.special.gamma(0.5*k)))*( (Expct_Chi(t, 1)) +((1-2*k)/6)*(Expct_Chi(t, 3))+(((3-4*k)*(1-4*k))/(120))*(Expct_Chi(t,5)) ) ))
'''

#def Expct_p_t_k(chi, k):
    #return
#print(py.special.gamma(0.5))


area = cumfunction(1,-1, 0, 1)


print(area)








################################################                 inizio      K-impact               ########################


k = [0.02 , 0.05 , 0.1 , 0.2 , 0.5 , 1]
ttt = np.logspace(0, 3.5, 250)
#ttt = np.linspace(1, 2000, 250)

delta_p_t_k_0=[]
delta_p_t_k_1=[]
delta_p_t_k_2=[]
delta_p_t_k_3=[]
delta_p_t_k_4=[]
delta_p_t_k_5=[]

for i in range (0, 250, 1):
    t = ttt[i]
    nu = 0.1

    chi0_k= ((np.random.normal(nu*np.sqrt(1), np.sqrt(1-nu**2), 100000)))
    chi_k_t= ((np.random.normal(nu*np.sqrt(t), np.sqrt(1-nu**2), 100000)))

    p_t_k_0_avg_0 = ((p_t_k_2(chi0_k, k[0]).sum())/100000)
    p_t_k_1_avg_0 = ((p_t_k_2(chi0_k, k[1]).sum())/100000)
    p_t_k_2_avg_0 = ((p_t_k_2(chi0_k, k[2]).sum())/100000)
    p_t_k_3_avg_0 = ((p_t_k_2(chi0_k, k[3]).sum())/100000)
    p_t_k_4_avg_0 = ((p_t_k_2(chi0_k, k[4]).sum())/100000)
    p_t_k_5_avg_0 = ((p_t_k_2(chi0_k, k[5]).sum())/100000)

    p_t_k_0_avg_t = ((p_t_k_2(chi_k_t, k[0]).sum())/100000)
    p_t_k_1_avg_t = ((p_t_k_2(chi_k_t, k[1]).sum())/100000)
    p_t_k_2_avg_t = ((p_t_k_2(chi_k_t, k[2]).sum())/100000)
    p_t_k_3_avg_t = ((p_t_k_2(chi_k_t, k[3]).sum())/100000)
    p_t_k_4_avg_t = ((p_t_k_2(chi_k_t, k[4]).sum())/100000)
    p_t_k_5_avg_t = ((p_t_k_2(chi_k_t, k[5]).sum())/100000)

    #print(p_t_k_0_avg_t)

    delta_p_t_k_0.append( p_t_k_0_avg_t - p_t_k_0_avg_0 )
    delta_p_t_k_1.append( p_t_k_1_avg_t - p_t_k_1_avg_0 )
    delta_p_t_k_2.append( p_t_k_2_avg_t - p_t_k_2_avg_0 )
    delta_p_t_k_3.append( p_t_k_3_avg_t - p_t_k_3_avg_0 )
    delta_p_t_k_4.append( p_t_k_4_avg_t - p_t_k_4_avg_0 )
    delta_p_t_k_5.append( p_t_k_5_avg_t - p_t_k_5_avg_0 )




    '''
    delta_p_t_k_0.append( (( (p_t_k_1(chi_k_t, k[0]) - p_t_k_1(chi0_k, k[0])).sum() ) /100000))
    delta_p_t_k_1.append( (( (p_t_k_1(chi_k_t, k[1]) - p_t_k_1(chi0_k, k[1])).sum() ) /100000))
    delta_p_t_k_2.append( (( (p_t_k_1(chi_k_t, k[2]) - p_t_k_1(chi0_k, k[2])).sum() ) /100000))
    delta_p_t_k_3.append( (( (p_t_k_1(chi_k_t, k[3]) - p_t_k_1(chi0_k, k[3])).sum() ) /100000))
    delta_p_t_k_4.append( (( (p_t_k_1(chi_k_t, k[4]) - p_t_k_1(chi0_k, k[4])).sum() ) /100000))
    delta_p_t_k_5.append( (( (p_t_k_1(chi_k_t, k[5]) - p_t_k_1(chi0_k, k[5])).sum() ) /100000))
    '''




'''
print(p_t_k_0_avg_0, "\n")
print(p_t_k_1_avg_0, "\n")
print(p_t_k_2_avg_0, "\n")
print(p_t_k_3_avg_0, "\n")
print(p_t_k_4_avg_0, "\n")
print(p_t_k_5_avg_0, "\n")



print(p_t_k_0_avg_t, "\n")
print(p_t_k_1_avg_t, "\n")
print(p_t_k_2_avg_t, "\n")
print(p_t_k_3_avg_t, "\n")
print(p_t_k_4_avg_t, "\n")
print(p_t_k_5_avg_t, "\n")
'''
#print(delta_p_t_k_0)
#print(ttt)


#############################################                      fine K-impact                                    #######################

#area= py.stats.norm.cdf(1, loc=0, scale=1)-py.stats.norm.cdf(-1, loc=0, scale=1)

area = cumfunction(1,-1, 0, 1)


print(area)














plt.figure(5)
plt.xscale('log')                             #plotto il grafico al variare dei k
plt.yscale('log')
plt.title("price impact for different k  [nu = 0.1]")
plt.ylabel("E[P_t -p_1] al variare di k")
plt.xlabel("t")
plt.ylim(0.0001, 100)
plt.xlim(0.8, 7000)
plt.errorbar(ttt, delta_p_t_k_0,  color="blue", fmt="-")     #plotto dati simulati con distribuzioni al variare di k
plt.errorbar(ttt, delta_p_t_k_1,  color="red", fmt="-")
plt.errorbar(ttt, delta_p_t_k_2,  color="salmon", fmt="-")
plt.errorbar(ttt, delta_p_t_k_3,  color="gray", fmt="-")
plt.errorbar(ttt, delta_p_t_k_4,  color="black", fmt="-")
plt.errorbar(ttt, delta_p_t_k_5,  color="orange", fmt="-")











plt.show()
