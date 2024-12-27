import math
import sympy
import numpy as np
from collections import namedtuple
import re

T = 6380        #Tara (T) 6380 kg VW (2023)
PL = 7920       #Peso Líquido (PL) 7920 kg VW (2023)
PBT = 14300     #Peso Bruto Total (PBT) 14300 kg VW (2023)
Q_bat = 105     #Capacidade da bateria 105 kWh VW (2023)
Afr = 4.45      #Área frontal (Afr) 4,45 m2 VW (2023)

fr = lambda v: 0.01 * (1 + 1.60934 * (v / 100) ) #Resistência ao rolamento (fr) fr(vel) - Donkers et al. (2020)
farr = 0.7       #Coeficiente de arrasto (farr) 0,7 - Fiori et al. (2021)
g = 9.81         #Aceleração da gravidade (g) 9,81 m/s2 Fiori et al. (2021)
p_ar = 1.255     #Densidade do ar (ρar) 1,225 kg/m3 Picard et al. (2008)
Nmotor = 0.95    #Eficiência do motor (ηmot) 95% - Pena et al. (2024)
Nbat = 0.97      #Eficiência da bateria (ηbat) 97% - Pena et al. (2024)
Ntransm = 0.95   #Eficiência da transmissão (ηtr) 96% - Pena et al. (2024)
Nconv = 0.90     #Eficiência dos conversores (ηconv) 90% - Pena et al. (2024)
Ncondutor = 0.90 #Eficiência do condutor (ηcondutor) 90% - Donkers et al. (2020)
Nglobal = Nmotor * Nbat * Ntransm * Nconv #Eficiência global do veículo

frb = lambda v: 1 - (np.exp(-v)) #Fator de frenagem regenerativa (frb) frb(vel) - Pena et al. (2024)
#Nrb = Eficiência do sistema regenerativo (ηrb) ηrb(frb, vel, η) - Pena et al. (2024)

gama_aux = 1.3  #Consumo dos sistemas auxiliares (γaux) 1,3 - O autor.

def E_total(m, ai, vi, teta_i, delta_T, printData=True):
    mi = T + m
    NPoints = len(ai)

    E = 0
    Etracao = 0
    Eregen = 0
    
    E_cinetica_total = 0
    E_gravitacional_total = 0
    E_arrasto_total = 0
    E_rolamento_total = 0
    
    for i in range(NPoints):
        Ftracao_i, Faceleracao_i, Farrasto_i, Fgravitacional_i,Frolamento_i = __forcas(mi, ai[i], vi[i], teta_i[i])

        Ecinetica = __Ecinetica(Faceleracao_i, vi[i], delta_T[i], ai[i])
        Earrasto_i = Farrasto_i * vi[i] * delta_T[i]
        Egravidade_i = __Egravidade(Fgravitacional_i, vi[i], delta_T[i], teta_i[i])
        E_rolamento_i = Frolamento_i * vi[i]*delta_T[i]
        
        Eregen = __Eregen(Faceleracao_i, vi[i], delta_T[i], ai[i])
        
        Eregen_i = (-1) * Eregen * 0.001
        Etracao_i = (\
            (Ecinetica)+\
            (Earrasto_i)+\
            (Egravidade_i)+\
            (E_rolamento_i)\
            ) * 0.001
        
        E_cinetica_total += Ecinetica * 0.001
        E_gravitacional_total += Egravidade_i * 0.001
        E_arrasto_total += Earrasto_i * 0.001
        E_rolamento_total += E_rolamento_i * 0.001
        
        Etracao += Etracao_i
        Eregen += Eregen_i
        
        if printData:
            print("================")
            print("No ponto", i, 
                  "\nE_cinetica = ", Ecinetica, "kWh", 
                  "\nE_arrasto = ", Earrasto_i, "kWh",
                  "\nE_gravitacional = ", Egravidade_i, "kWh",
                  "\nE_rolamento = ", E_rolamento_i, "kWh")
            print("\nE_tracao = ", Etracao, "kWh", ", E_regen_exp = ", Eregen, "kWh")
        
    E_k = (gama_aux / Nglobal) * E_cinetica_total
    E_g = (gama_aux / Nglobal) * E_gravitacional_total
    E_arr = (gama_aux / Nglobal) * E_arrasto_total
    E_atrito = (gama_aux / Nglobal) * E_rolamento_total
    
    E = (gama_aux / Nglobal) * Etracao - (Eregen * Nglobal)
    
    return E, E_k, E_g, E_arr, E_atrito

def __forcas(mi, ai, vi, teta_i):
    Faceleracao_i = mi * ai
    Farrasto_i = (1 / 2) * p_ar * farr * Afr * (vi**2)
    Fgravitacional_i = mi * g * math.sin(teta_i)
    Frolamento_i = mi * g * math.cos(teta_i) * fr(vi)
    
    Ftracao_i = Faceleracao_i + Farrasto_i + Fgravitacional_i + Frolamento_i

    return Ftracao_i, Faceleracao_i, Farrasto_i, Fgravitacional_i, Frolamento_i
    
def __Ecinetica(Faceleracao_i, vi, delta_T, ai, tipo = "+"):
    if ai >= 0:
        Ecinetica = Faceleracao_i * vi * delta_T
    else:
        Ecinetica = 0
    return Ecinetica

def __Eregen(Faceleracao_i, vi, delta_T, ai, tipo = "+"):
    if ai < 0:
        Eregen = Faceleracao_i * vi * delta_T * frb(vi)
    else:
        Eregen = 0
    return Eregen

def __Egravidade(Fgravitacional_i, vi, delta_T, teta_i):
    if teta_i > 0: 
        Egravidade = Fgravitacional_i * vi * delta_T
    else:
        Egravidade = 0
    return Egravidade