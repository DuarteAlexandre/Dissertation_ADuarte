# Importar dados
import pandas as pd

df = pd.read_excel("DELIVERIES.xlsx", "deliveries_electric_vf")

# Referneicas de COletas de dados da API
from src.MP import MP
from src.MYrequest2 import MYrequest
from src.getDistance.getRealDistance import getRealDistance
from srcTeste.getRef import getRef
from random import shuffle
from glob import glob

keys = ["34847c15-fb36-436a-aa39-d824540c952c","6977925f-018c-4cd0-944d-a39c3835316d", "661640f2-70d9-4e0f-ac20-f79223969d80"]     
idx = 2
def make_api_request(data):
    return MYrequest(data, getRealDistance)#, key=keys[idx]

def tratamento(resps, refs):
    dists = []
    times = []
    geometrys = []
    for c, resp in enumerate(resps):
        ref = refs[c]
        for idx in range(1, len(ref)):
            i = ref[idx-1]
            j = ref[idx]

            try:
                dists.append({"origin": i, "destination": j, "value": resp[0][idx-1]})
            except:
                dists.append({"origin": i, "destination": j, "value": "Error"})
            try:
                times.append({"origin": i, "destination": j, "value": resp[1][idx-1]})
            except:
                times.append({"origin": i, "destination": j, "value": "Error"})
            try:
                for geo in resp[2][idx-1]:
                    geometrys.append({"origin": i, "destination": j, "latlng": geo["latlng"],"value": geo["distance"]})
            except:
                geometrys.append({"origin": i, "destination": j, "latlng":"Error","value": "Error"})

            #dists.append({(i,j): resp[0][idx-1]})
        #if len(resp[0])>0:
        #    print(resp[0])
        #    dist = 
        #    dists.append(dist)
        #    time = resp[1][0]
        #    times.append(time)
        #    geometrys += resp[2][0]
    
    return dists, times, geometrys

pointOSRM = lambda latlng: "{},{}".format(latlng[1], latlng[0]) 
pointsOSRM = lambda LL: ";".join(pointOSRM(pontos[i]) for i in LL)

modo = "OSRM"
points = lambda LL: pointsOSRM(LL)

if __name__ == '__main__':
    datas = df["tour_date"].unique()
    # Data de Referencia
    Datas_Relevantes = []
    for d in range(len(datas)):
        df_avaliado = df[df["tour_date"] == datas[d]]

        if len(df_avaliado)>=200: Datas_Relevantes.append(datas[d])


    for d, data in enumerate(Datas_Relevantes):
        if d>0:continue
        df_avaliado = df[df["tour_date"] == data]
        df_avaliado = df_avaliado.reset_index()

        # Pontos de Refernecia
        
        pontos = {
            df_avaliado.loc[i, "client_id"]: (df_avaliado.loc[i, "poc_latitude"], df_avaliado.loc[i, "poc_longitude"]) for i in range(len(df_avaliado))
        }
        pontos["CD"] = (-23.495497928383347, -46.76044616284469)


        distancias = list()
        tempos = list()
        geometrys = list()


        #DADOS = list(map(lambda d: (d["TIME (MIN)"], d["VEL_KMH"], d["a (m/s2)"], d["ID"], d["massa"], d["VEL"], d["VOLT"], d["ODOM"]) ,dados))
        DADOS = list(map(lambda d: (0, 0, 0, d, 0, 0, 0, 0) ,pontos))

        
        ArcsEstudados = [(i, j) for i in pontos for j in pontos if i !=j]
        shuffle(ArcsEstudados)
        
        print("INICIANDO PROCESSAMENTO")
        ref = [-1]
        input_data = []
        refs = []
        nP = 10

        while len(ref):                    
            ref, ArcsEstudados = getRef(pontos, ArcsEstudados.copy(), ref = [], n=30)
            
            if len(ref)==0: continue
            
            input_data.append((points(ref)))
            refs.append(ref)

            if len(input_data)>=nP:
                resps = MP(input_data, make_api_request)
                ds, ts, gs = tratamento(resps, refs)
                distancias+=ds
                tempos+=ts
                geometrys += gs
                input_data = []

            print(len(ArcsEstudados))
        if len(input_data):
            resps = MP(input_data, make_api_request)
            ds, ts, gs = tratamento(resps, refs)
            distancias+=ds
            tempos+=ts
            geometrys += gs
        print(f"Salvando dados da data {data}")
        #DadosInstancias
        df0 = pd.DataFrame([{"ID": p, "latitude": pontos[p][0], "longitude": pontos[p][1]} for p in pontos])
        df1 = pd.DataFrame(distancias)
        df2 = pd.DataFrame(tempos)
        df3 = pd.DataFrame(geometrys)
        # Usando o ExcelWriter, cria um doc .xlsx, usando engine='xlsxwriter'
        with pd.ExcelWriter(f".\DadosInstancias/tabelas_{str(data)[:10]}.xlsx") as writer:
            df0.to_excel(writer, sheet_name='Pontos')
            df1.to_excel(writer, sheet_name='Tabela Distancia')
            df2.to_excel(writer, sheet_name='Tabela Tempo')
            df3.to_excel(writer, sheet_name='Tabela Geometry')
        

        