
import math
#import symbol

T = 6380        #Tara (T) 6380 kg VW (2023)
PL = 7920       #Peso Líquido (PL) 7920 kg VW (2023)
PBT = 14300     #Peso Bruto Total (PBT) 14300 kg VW (2023)
Q_bat = 105     #Capacidade da bateria 105 kWh VW (2023)
K_larg = 2.035  #Largura máxima dianteira (K) 2,035 m VW (2023)
V_diant = 0.214 #Vão livre dianteiro 0,213 m VW (2023)
H = 2.398       #Altura 2,398 m VW (2023)
Afr = 4.45      #Área frontal (Afr) 4,45 m2 VW (2023)

fr = lambda v: 0.01 * (1+1.60934*v / 100) #Resistência ao rolamento (fr) fr(vel) - Donkers et al. (2020)
farr = 0.7      #Coeficiente de arrasto (farr) 0,7 - Fiori et al. (2021)
g=9.81          #Aceleração da gravidade (g) 9,81 m/s2 Fiori et al. (2021)
p_ar = 1.255    #Densidade do ar (ρar) 1,225 kg/m3 Picard et al. (2008)
Nmotor = 0.95   #Eficiência do motor (ηmot) 95% - Pena et al. (2024)
Nbat = 0.97     #Eficiência da bateria (ηbat) 97% - Pena et al. (2024)
Ntransm = 0.95  #Eficiência da transmissão (ηtr) 96% - Pena et al. (2024)
Nconv = 0.90    #Eficiência dos conversores (ηconv) 90% - Pena et al. (2024)
Ncondutor = 0.90#Eficiência do condutor (ηcondutor) 90% - Donkers et al. (2020)
#Eficiência do sistema regenerativo (ηrb) ηrb(frb, vel, η) - Pena et al. (2024)
#Fator de frenagem regenerativa (frb) frb(vel) - Pena et al. (2024)
gama_aux = 1.3  #Consumo dos sistemas auxiliares (γaux) 1,3 - O autor.


Nglobal = Nmotor * Nbat * Ntransm * Nconv

def E_total(mi, ai, vi, teta_i, delta_T):
    NPoints = len(ai)

    E = 0
    Etracao = 0
    Eregen = 0
    for i in range(NPoints):
        Ftracao_i, Faceleracao_i, Farrasto_i, Fgravitacional_i,Frolamento_i = __forcas(mi, ai[i], vi[i], teta_i[i])

        Ecinetica = __Ecinetica(Faceleracao_i, vi[i], delta_T[i], ai[i])
        Earrasto_i = Farrasto_i * vi[i] * delta_T[i]
        Egravidade_i = __Egravidade(Fgravitacional_i, vi[i], delta_T[i])
        E_rolamento_i = Frolamento_i * vi[i]*delta_T[i]
        
        Eregen_i = -1 * (Ecinetica if ai[i]<0 else 0) * Ncondutor * 0.001
        Etracao_i = (\
            (Ecinetica if ai[i]>=0 else 0)+\
            Earrasto_i+\
            (Egravidade_i if teta_i[i]>0 else 0)+\
            E_rolamento_i\
            ) * 0.001

        Etracao += Etracao_i
        Eregen += Eregen_i

        print("================")
        print(i, Ecinetica, Earrasto_i, Egravidade_i, E_rolamento_i)
        print(Etracao, Eregen)
        

    E = gama_aux / Nglobal * Etracao - Eregen * Nglobal
    
    return E

    # c (kWH / KG) * m (kg)

def __forcas(mi, ai, vi, teta_i):
    Faceleracao_i = mi * ai
    Farrasto_i = 1 / 2 * p_ar * farr * Afr * vi**2
    Fgravitacional_i = mi * g * math.sin(teta_i)
    Frolamento_i = mi*g* math.cos(teta_i) * fr(vi)
    
    Ftracao_i = Faceleracao_i + Farrasto_i + Fgravitacional_i + Frolamento_i

    return Ftracao_i, Faceleracao_i, Farrasto_i, Fgravitacional_i, Frolamento_i
    
def __Ecinetica(Faceleracao_i, vi, delta_T, ai, tipo = "+"):
    Ecinetica = Faceleracao_i * vi * delta_T 
    return Ecinetica
def __Egravidade(Fgravitacional_i, vi, delta_T):
    Egravidade = Fgravitacional_i * vi * delta_T
    return Egravidade




import sympy

mi = sympy.symbols("m")

#mi = 10000
vi = [0, 15, 20, 15, 0] #[15/3.6 for i in range(5)]
vi = [vi[i]/3.6 for i in range(5)]
teta_i = [0] + [math.radians(1 * 45 / 100) for i in range(1,5)]
delta_T = [0] + [1*60 / 3600 for i in range(1, 5)]
ai = [0] + [(vi[i]-vi[i-1])/(delta_T[i]*3600) for i in range(1, 5)]

E = E_total(mi, ai, vi, teta_i, delta_T)
#PERCENT = E / Q_bat * 100
#print("Energia: ", E, PERCENT, "%")
#print("DIST", [0] + [(vi[i]**2 - vi[i-1]**2) / (2 * ai[i]) for i in range(1, 5)])
#print("vi", vi)
#print("ai", ai)
#print("teta_i", teta_i)
#print("Nglobal", Nglobal)
print("================")
print(E)
