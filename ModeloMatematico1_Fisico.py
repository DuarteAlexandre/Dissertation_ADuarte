import gurobipy as gp

def ModeloMatematico(C, N, A, K, Q, d_i, SOH, Bmax, Bmin, Error, e_FIXO_ij, e_ij, rotainicial = {}, S=[]):

    m = gp.Model()
    m.setParam("MIPGap", 0.05)
    time = 60*5
    m.setParam("TimeLimit", time)
        
    f_ijk = m.addVars(N, N, K, name="f_ijkt") # MASSA (KG) AO SAIR DE i PARA j.
    x_ijk = m.addVars(N, N, K, vtype = gp.GRB.BINARY, name="x_ijkt") # BINARIA 1 SE ARCO i,j É UTILIZADO
    g_ijk = m.addVars(N, N, K, name="g_ijkt") # NIVEL DA BATERIA AO SAIR DE i PARA j, EM KWH

    # OBJETIVO MINIMIZAR O TAMANHO DA FROTA

    m.setObjective(
        gp.quicksum(x_ijk[0,j,k] for j in N for k in K),
        sense=gp.GRB.MINIMIZE
    )
    
    # (1) GARANTE QUE TODOS OS CLIENTE DEVEM SER ATENDIDOS
    m.addConstrs(
        gp.quicksum(x_ijk[i,j,k] for k in K for i in N if (i,j) in A ) == 1

        for j in C
    )

    # (2) GARANTE CONSERVAÇÃO DO FLUXO NA REDE
    m.addConstrs(
        gp.quicksum(x_ijk[j,i,k]  for j in N if i!=j) -
        gp.quicksum(x_ijk[i,j,k]  for j in N if i!=j) == 0

        for i in N
        for k in K
    )

    # (3) GARANTE UNICA ROTA POR VEICULO
    m.addConstrs(
        gp.quicksum(x_ijk[0,j,k]  for j in C) <= 1

        for k in K
    )

    # (4) GARANTE CONSERVAÇÃO DO FLUXO DE MASSA NA ROTA e RESTRINGE SUBCICLOS.
    m.addConstrs(
        gp.quicksum(f_ijk[j,i,k] for k in K for j in N if (j,i) in A ) -
        gp.quicksum(f_ijk[i,j,k] for k in K for j in N if (i,j) in A ) == d_i[i]

        for i in C
    )

    # (5) CAPACIDADE DO VEICULO
    m.addConstrs(
        f_ijk[i,j,k] <= Q * x_ijk[i,j,k]

        for k in K
        for i,j in A
    )

    # (6) GARANTE CONSERVAÇÃO DO GASTO DE ENERGIA
    m.addConstrs(
        gp.quicksum(g_ijk[j,i,k] for j in N if i!=j) -
        gp.quicksum(g_ijk[i,j,k] for j in N if i!=j) == 
        gp.quicksum(e_FIXO_ij[j, i] * x_ijk[j ,i, k] + e_ij[j, i] * f_ijk[j, i, k] for j in N if (j,i) in A)

        for i in C
        for k in K
    )

    # (7) GARANTE CAPACIDADE MAXIMA DE 100 kWh DO NIVEL DE ENERGIA
    m.addConstrs(
        g_ijk[i,j,k] <= Bmax * x_ijk[i,j,k] * SOH

        for k in K
        for i,j in A
    )

    # (8) GARANTE BATERIA 100 kWh AO SAIR DO CD
    m.addConstrs(
        g_ijk[0,j,k] == Bmax * x_ijk[0,j,k] * SOH

        for k in K
        for j in N if (0,j) in A
    )

    # (9) GARANTE ENERGIA MINIMA de 10kWh AO CHEGAR NO CD
    m.addConstrs(
        g_ijk[i,0,k] >= (Bmin + Error) * x_ijk[i,0,k] 

        for k in K
        for i in N if (i, 0) in A
    )

    print("RESTRICOES DE CORTE DE HIPERPLANO")
    m.addConstrs(
        gp.quicksum(x_ijk[i,j,k] for i in S[idx_s] for j in S[idx_s] for k in K  if i!=j) <= len(S[idx_s])-1
        for idx_s in range(len(S)-1,-1,-1)
    )

    m.update()
    initial_solution = {}
    if rotainicial != {}:
        
        for i,j,k in rotainicial["f_ijk"]:
            initial_value = rotainicial["f_ijk"][i,j,k] # Defina sua lógica para o valor inicial
            initial_solution[f_ijk[i,j,k]] = initial_value

        for i,j,k in rotainicial["x_ijk"]:            
            initial_value = rotainicial["x_ijk"][i,j,k] # Defina sua lógica para o valor inicial
            initial_solution[x_ijk[i,j,k]] = initial_value
       
        for i,j,k in rotainicial["g_ijk"]:
            initial_value = rotainicial["g_ijk"][i,j,k] 
            initial_solution[g_ijk[i,j,k]] = initial_value
            
    for var in initial_solution:
        var.start = initial_solution[var]
        
            
    m.optimize()
    
    if m.status == gp.GRB.Status.INFEASIBLE or not m.objVal < gp.GRB.INFINITY:
        print('Modelo é inviável.')
        m.setParam("TimeLimit", 9999999999)
        m.setParam('SolutionLimit', 1)  # Para parar após encontrar uma solução viável
        m.optimize()

    gap = m.MIPGap

    FleetSize = sum(x_ijk[0,j,k].X for j in N for k in K)

    return m, x_ijk, f_ijk, g_ijk, gap, FleetSize