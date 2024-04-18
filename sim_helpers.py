import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import networkx as nx
import time
import numpy.linalg as lin
import random
from matplotlib.legend_handler import HandlerTuple


def create_scale_free_economic_network(num_economies, m):
    """
    Function will create a scale free graph with a master economy selected by the node with the most links
    Arguments:
        num_economies : The number of nodes you want in the network
        m : The number of new connections each node should get each iteration
    """

    # Create the scale free graph
    G = nx.barabasi_albert_graph(num_economies, m)
    
    degrees = dict(G.degree())  # Dictionary mapping nodes to their degrees
    max_degree_node = max(degrees, key=degrees.get)  # Get the node with the highest degree
    max_degree = degrees[max_degree_node] # Get the highest number of degrees
    
    # Set attributes for Master economy
    for node in G._node:
        if node == max_degree_node:
            G.nodes[node]['name'] = 'Master'
        else:
            G.nodes[node]['name'] = 'sat'

    return G, max_degree_node, max_degree

def initialize_sim(num_economies, num_edges):

    """
    Function will create a scale free graph with a master economy selected by the node with the most links.
    This function will use create_scale_free_economic_network to display the information of the graph created.
    Arguments:
        num_economies : The number of nodes you want in the network
        num_edges : The number of new connections each node should get each iteration
    """

    # Create the graph and select the master economy as the one with the most degrees
    G, max_d_node, max_d = create_scale_free_economic_network(num_economies, num_edges)

    # Print graph information
    print('Economy', max_d_node, 'is the master econmy.')
    print('The Master Economy has degree', max_d)
    print('The number of total links is:', G.number_of_edges())

    # Calculate the average degree of the network
    average_degree = sum(dict(G.degree()).values())  / len(G)
    print('Average degree of the network:', average_degree)

    # set up the labels for visualization
    labels = {}
    ind = 0
    for node in G._node:
        if node == max_d_node:
            labels[ind] = 'Master'
        else:
            labels[ind] = str(ind)
        ind+=1

    # loop over each node and set the various parameters up
    for loop in G._node:
        G._node[loop]['index'] = ind
        G._node[loop]['link'] = list(nx.neighbors(G,loop))
        G._node[loop]['numlink'] = len(list(nx.neighbors(G,loop)))
    # Draw the network
    plt.figure(figsize=(10,6))
    pos = nx.spring_layout(G)  # Layout for positioning nodes
    nx.draw_networkx_nodes(G, pos, node_color="cornflowerblue")
    ind = 0
    direct_nodes = []
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'forestgreen'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                direct_nodes.append(cindex)
        ind += 1
    
    nx.draw_networkx_nodes(G, pos, nodelist=direct_nodes, node_color="forestgreen")
    nx.draw_networkx_nodes(G, pos, nodelist=[max_d_node], node_color="orange")
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels)

    # Show plot
    plt.title('Economic System Network')
    plt.show()

    return G

def van_der_pol(G, u0, t):
    """
    Function will solve the ODE for coupled van der pol oscillators (with Master oscillator not affected)

    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just couple all nodes normally
        u0 : The initial position and velocity of each node in a 1-D array [x1,v1,x2,v2,...]
        t : The array time stepped interval
    """

    def f(u,t):
        temp = np.zeros(len(u)) # set up a tempoorary storage for positions and velocities
        ind = 0 # dummy index

        # iterate over each node
        for i in G._node:
            linksz = G._node[i]['numlink'] # check the number of links each node has
            mu = G._node[i]['mu'] # identify the parameter 'mu'

            # origional flow equations (van der pol)
            temp[ind*2] = u[ind*2 + 1]
            temp[ind*2 + 1] = mu*(1 - u[ind*2]**2) * u[ind*2 + 1] - u[ind*2]
            
            # add coupling only if we are not the master (master evolves normally)
            if G._node[i]['name'] != 'Master':
                for gloop in range(linksz):
                    cindex = G._node[i]['link'][gloop]
                    indx = G._node[cindex]['index']
                    g = G._node[i]['coupling'][gloop]

                    temp[ind*2 + 1] = temp[ind*2 + 1] - g * (u[ind*2] - u[indx*2])
                
            ind += 1

        return temp
    
    # Solve for the trajectories
    U = integrate.odeint(f, u0, t)

    return U


def ISLM_model(G, u0, t):
    """
    Function will solve the ODE for coupled IS-LM models (with Master oscillator not affected)

    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just couple all nodes normally
        u0 : The initial interest rate (r, dr/dt) and real income [GDP] (g, dg/dt) of each node in a 1-D array
             [r1, g1, r2, g2,...]
        t : The array time stepped interval
    """

    def f(u,t):
        temp = np.zeros(len(u)) # set up a tempoorary storage for positions and velocities
        ind = 0 # dummy index

        # iterate over each node
        for i in G._node:
            linksz = G._node[i]['numlink'] # check the number of links each node has
            mu = G._node[i]['mu'] # identify the parameter 'mu'
            alpha = G._node[i]['alpha']
            beta = G._node[i]['beta']
            c = G._node[i]['c']
            l = G._node[i]['l']
            k = G._node[i]['k']
            v = G._node[i]['v']
            m = G._node[i]['m']

            # origional flow equations (IS_LM)
            #r ̇ = β (kg − vr − m0)
            #g ̇ = α (Cg − ur + G) 
            
            temp[ind*2] = beta * (k * u[ind*2+1] - v * u[ind*2] + m)
            temp[ind*2+1] = alpha * (c * u[ind*2+1] - mu * u[ind*2] + l)
            
            
            # add coupling only if we are not the master (master evolves normally)
            if G._node[i]['name'] != 'Master':
                for gloop in range(linksz):
                    cindex = G._node[i]['link'][gloop]
                    indx = G._node[cindex]['index']
                    g = G._node[i]['coupling'][gloop]

                    # we could add a different coupling strength to each of the variables?
                    temp[ind*2] = temp[ind*2] - g * (u[ind*2] - u[indx*2])
                    temp[ind*2 + 1] = temp[ind*2 + 1] - g * (u[ind*2+1] - u[indx*2+1])

                
            ind += 1

        return temp
    
    # Solve for the trajectories
    U = integrate.odeint(f, u0, t)

    return U

def kaldor_model(G, u0, t):
    """
    Function will solve the ODE for coupled kaldor buisiness cycle models (with Master oscillator not affected)

    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just couple all nodes normally
        u0 : The initial Capital Stock (K, dK/dt) and Output (Y, dY/dt) of each node in a 1-D array
             [K1, Y1, K2, Y2,...]
        t : The array time stepped interval
    """

    def f(u,t):
        temp = np.zeros(len(u)) # set up a tempoorary storage for positions and velocities
        ind = 0 # dummy index

        # iterate over each node
        for i in G._node:
            linksz = G._node[i]['numlink'] # check the number of links each node has
            mu = G._node[i]['mu'] # identify the parameter 'mu'
            alpha = G._node[i]['alpha']
            beta = G._node[i]['beta']
            c = G._node[i]['c']
            l = G._node[i]['l']
            k = G._node[i]['k']
            v = G._node[i]['v']
            m = G._node[i]['m']

            # origional flow equations (Kaldor)
            # x' = -k * x + m tanh(l * v) + SUM(i)[g * (x - x_i)]
            # v' = -alpha * x + beta * v - c * v^3 + SUM(i)[g * (v - v_i)]
            
            
            temp[ind*2] = -k * u[ind*2] + m * np.tanh(l * u[ind*2+1])
            temp[ind*2+1] = -alpha * u[ind*2] + beta * u[ind*2+1] - c * (u[ind*2+1])**3
            
            
            # add coupling only if we are not the master (master evolves normally)
            if G._node[i]['name'] != 'Master':
                for gloop in range(linksz):
                    cindex = G._node[i]['link'][gloop]
                    indx = G._node[cindex]['index']
                    g = G._node[i]['coupling'][gloop]

                    # we could add a different coupling strength to each of the variables?
                    temp[ind*2] = temp[ind*2] - g * (u[ind*2] - u[indx*2])
                    temp[ind*2 + 1] = temp[ind*2 + 1] - g * (u[ind*2+1] - u[indx*2+1])
                
            ind += 1

        return temp
    
    # Solve for the trajectories
    U = integrate.odeint(f, u0, t)

    return U

# def run_simulation(G,  total_time, t_divisions, mu=1, g=1, x0=1, v0=1, ode_func = van_der_pol):
#     """
#     Function will run the simulation (calling the van_der_pol function - other can be specified)
#     It takes your graph and adds random parameters to all of the nodes (except the master which can be specified)

#     Arguments:
#         G : The graph (networkx) with 1 node having the name 'Master'
#             If there is no node named 'Master' then the function will just couple all nodes normally
#         total_time : The total time that the simulation should span
#         t_divisions: The number of time step divisions that you want
#         mu : non-linearity and strength of damping
#         g : The coupling strength from the Master to other nodes
#         x0 : The initial position of the master economy
#         v0 : The initial velocity of the master economy
#     """

#     N = G.number_of_nodes() # number of nodes in the graph
#     lnk = np.zeros(shape = (N,), dtype=int) # initialize an array to hold info on which nodes are linked
#     ind = 0 # dummy index
    
#     # this 1-D array will store the initial position and velocity of each node sequentially
#     u0 = np.zeros(shape = (N * 2))
    
#     # loop over each node and set the various parameters up
#     for loop in G._node:
#         G._node[loop]['index'] = ind
#         G._node[loop]['link'] = list(nx.neighbors(G,loop))
#         G._node[loop]['numlink'] = len(list(nx.neighbors(G,loop)))
#         lnk[ind] = len(list(nx.neighbors(G,loop)))
#         G._node[loop]['coupling'] = np.zeros(shape=(lnk[ind],))
#         for linkloop in range (lnk[ind]):
#             G._node[loop]['coupling'][linkloop] = random.random()
#         G._node[loop]['mu'] = random.random()
#         u0[ind*2 + 0] = random.random()*5
#         u0[ind*2 + 1] = random.random()*5

#         # set the master initial conditions so we know what they are
#         if G._node[loop]['name'] == 'Master':
#             G._node[loop]['mu'] = mu
#             u0[ind*2 + 0] = x0
#             u0[ind*2 + 1] = v0
#             for linkloop in range (lnk[ind]):
#                 G._node[loop]['coupling'][linkloop] = g
#         ind+=1

#     # set the time interval with time steps
#     t = np.linspace(0, total_time, t_divisions)

#     # run the ODE solver
#     U = ode_func(G, u0, t)

#     return U, t

def run_simulation(G, total_time, t_divisions, x0=1, v0=1, g=1, param_perturb = 0.1,
                   mu=1, alpha=1, beta=1, c=1, l=1, k=1, v=1, m=1,ode_func = van_der_pol):
    """
    Function will run the simulation (calling the van_der_pol function - other can be specified)
    It takes your graph and adds random perturbation parameters to all of the nodes off of the 
    values of the master parameters (which are specified)

    Arguments:
        The arguments will depend on which model you are choosing and they will be used accordingly in the function call
        Although all parameters are added to the nodes, only certain ones may be used for different simulations.
        G : The graph (networkx) with 1 node having the name 'Master'
            If there is no node named 'Master' then the function will just couple all nodes normally
        total_time : The total time that the simulation should span
        t_divisions: The number of time step divisions that you want
        g : The coupling strength from the Master to other nodes (and base coupling from satellite nodes)
        param_perturb : a small perturbation order added (positive or negative) to each of the parameters (random)

        ###########
        van-der-pol
            Flow: (last term is the coupling to other nodes)
                x = v
                v = mu*(1 - x^2) * v - x - SUM(i)[g * (x - x_i)]
        x0 : The initial position of the master economy (base for satellite nodes)
        v0 : The initial velocity of the master economy (base for satellite nodes)
        mu : non-linearity and strength of damping (base for satellite nodes)

        ###########
        IS-LM model
            Flow: (last term is the coupling to other nodes)
                x' = beta * (k * v - v * x + m) + SUM(i)[g * (x - x_i)]
                v' = alpha * (c * v - mu * x + l) + SUM(i)[g * (v - v_i)]
        x0 : The initial Interest Rate of the master economy (base for satellite nodes)
        v0 : The initial Real Income (GDP) of the master economy (base for satellite nodes)
        mu : Suppressed investment in response to interest rate r (base for satellite nodes)
        l : Autonomous (government) expenditure (base for satellite nodes)
        c = p (1 − T ) + (i − 1) : Define full value when entering into model
            - p : Propensity to consume
            - T : Tax rate
            - i : Increased investment in response to income
        k : Enhanced demand for money in response to income y
        v : Suppressed demand for money in response to interest rate r
        alpha : positive rate for interest rate
        beta : positive rate for real income

        ###########
        Kaldor Buisiness cycle model
            Flow: (last term is the coupling to other nodes)
                v' = -alpha * x + beta * v - c * v^3 + SUM(i)[g * (v - v_i)]
                x' = -k * x + m tanh(l * v) + SUM(i)[g * (x - x_i)]
        x0 : The initial Capital Stock of the master economy (base for satellite nodes)
        v0 : The initial Output of the master economy (base for satellite nodes)
        alpa : 
        beta : 
        c : 
        k : 
        m : 
        l : 

    """

    N = G.number_of_nodes() # number of nodes in the graph
    lnk = np.zeros(shape = (N,), dtype=int) # initialize an array to hold info on which nodes are linked
    ind = 0 # dummy index
    
    # this 1-D array will store the initial position and velocity of each node sequentially
    u0 = np.zeros(shape = (N * 2))
    
    # loop over each node and set the various parameters up
    for loop in G._node:
        G._node[loop]['index'] = ind
        G._node[loop]['link'] = list(nx.neighbors(G,loop))
        G._node[loop]['numlink'] = len(list(nx.neighbors(G,loop)))
        lnk[ind] = len(list(nx.neighbors(G,loop)))
        G._node[loop]['coupling'] = np.zeros(shape=(lnk[ind],))
        for linkloop in range (lnk[ind]):
            G._node[loop]['coupling'][linkloop] = g + (random.random() - random.random()) * param_perturb
        G._node[loop]['mu'] = mu + (random.random() - random.random()) * param_perturb
        G._node[loop]['alpha'] = alpha + (random.random() - random.random()) * param_perturb
        G._node[loop]['beta'] = beta + (random.random() - random.random()) * param_perturb
        G._node[loop]['c'] = c + (random.random() - random.random()) * param_perturb
        G._node[loop]['l'] = l + (random.random() - random.random()) * param_perturb
        G._node[loop]['k'] = k + (random.random() - random.random()) * param_perturb
        G._node[loop]['v'] = v + (random.random() - random.random()) * param_perturb
        G._node[loop]['m'] = m + (random.random() - random.random()) * param_perturb
        
        u0[ind*2 + 0] = random.random()
        u0[ind*2 + 1] = random.random()
        # can add additional factor of 10 if we are using ISLM_model
        if ode_func == van_der_pol:
            u0[ind*2 + 0]*=5
            u0[ind*2 + 1]*=5
        if ode_func == ISLM_model:
            u0[ind*2 + 0]*=50
            u0[ind*2 + 1]*=50
        

        # set the master initial conditions so we know what they are
        if G._node[loop]['name'] == 'Master':
            G._node[loop]['mu'] = mu
            G._node[loop]['alpha'] = alpha
            G._node[loop]['beta'] = beta
            G._node[loop]['c'] = c
            G._node[loop]['l'] = l
            G._node[loop]['k'] = k
            G._node[loop]['v'] = v
            G._node[loop]['m'] = m
            u0[ind*2 + 0] = x0
            u0[ind*2 + 1] = v0
            for linkloop in range (lnk[ind]):
                G._node[loop]['coupling'][linkloop] = g
        ind+=1

    # set the time interval with time steps
    t = np.linspace(0, total_time, t_divisions)

    # run the ODE solver
    U = ode_func(G, u0, t)

    return U, t



    ##################### CHAOS ####################

def chaos_model(G, u0, t):
    """
    Function will solve the ODE for coupled chaotic buisiness cycle models (with Master oscillator not affected)

    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just couple all nodes normally
        u0 : The sales (y, dy/dt), Output (V, dV/dt), and inventory stocks (B, dB/dt) of each node in a 1-D array
             [y1, V1, B1, y2, V2, B2, ...]
        t : The array time stepped interval
    """

    def f(u,t):
        temp = np.zeros(len(u)) # set up a tempoorary storage for positions and velocities
        ind = 0 # dummy index

        # iterate over each node
        for i in G._node:
            linksz = G._node[i]['numlink'] # check the number of links each node has
            # mu = G._node[i]['mu'] # identify the parameter 'mu'
            # alpha = G._node[i]['alpha']
            # beta = G._node[i]['beta']
            c = G._node[i]['c']
            l = G._node[i]['l']
            k = G._node[i]['k']
            # v = G._node[i]['v']
            m = G._node[i]['m']

            # origional flow equations (Chaotic Buisiness cycle)
            # V' = c_3 * B − c_2 * V − c_1 * y
            # B' = r/c_3 * y * ( 1 − y )
            # y' = V
            # let c_3 -> k, c_2 -> m, c_1 -> l, r -> c
            
            temp[ind*2+1] = k * u[ind*2+2] - m * u[ind*2+1] - l * u[ind*2]
            temp[ind*2+2] = c / k * u[ind*2] * (1 - u[ind*2])
            temp[ind*2] = u[ind*2+1]

            # add coupling only if we are not the master (master evolves normally)
            if G._node[i]['name'] != 'Master':
                for gloop in range(linksz):
                    cindex = G._node[i]['link'][gloop]
                    indx = G._node[cindex]['index']
                    g = G._node[i]['coupling'][gloop]

                    # adding a different coupling strength to each of the variables is possible
                    temp[ind*2] = temp[ind*2] - g * (u[ind*2] - u[indx*2])
                    temp[ind*2+1] = temp[ind*2+1] - g * (u[ind*2+1] - u[indx*2+1])
                    temp[ind*2+2] = temp[ind*2+2] - g * (u[ind*2+2] - u[indx*2+2])

            ind += 1

        return temp
    
    # Solve for the trajectories
    U = integrate.odeint(f, u0, t)

    return U

def run_chaos_simulation(G, total_time, t_divisions, x0=1, v0=1, y0=1, g=1, param_perturb = 0.1,
                   mu=1, alpha=1, beta=1, c=1, l=1, k=1, v=1, m=1,ode_func = van_der_pol):
    """
    Function will run the simulation (calling the van_der_pol function - other can be specified)
    It takes your graph and adds random perturbation parameters to all of the nodes off of the 
    values of the master parameters (which are specified)

    Arguments:
        The arguments will depend on which model you are choosing and they will be used accordingly in the function call
        Although all parameters are added to the nodes, only certain ones may be used for different simulations.
        G : The graph (networkx) with 1 node having the name 'Master'
            If there is no node named 'Master' then the function will just couple all nodes normally
        total_time : The total time that the simulation should span
        t_divisions: The number of time step divisions that you want
        g : The coupling strength from the Master to other nodes (and base coupling from satellite nodes)
        param_perturb : a small perturbation order added (positive or negative) to each of the parameters (random)

        Chaotic Buisiness cycle model
            Flow: (last term is the coupling to other nodes)
                V' = c_3 * B - c_2 * V - c_1 * y
                B' = r/c_3 * y * ( 1 - y )
                y' = V
                let c_3 -> k, c_2 -> m, c_1 -> l, r -> c
                let V -> v, B -> y, y -> x
        x0 : The initial sales of the master economy (base for satellite nodes)
        v0 : The initial output of the master economy (base for satellite nodes)
        y0 : The initial inventory stocks of the master economy (base for satellite nodes)
        c : 
        k : 
        m : 
        l : 

    """

    N = G.number_of_nodes() # number of nodes in the graph
    lnk = np.zeros(shape = (N,), dtype=int) # initialize an array to hold info on which nodes are linked
    ind = 0 # dummy index
    
    # this 1-D array will store the initial position and velocity of each node sequentially
    u0 = np.zeros(shape = (N * 3))
    
    # loop over each node and set the various parameters up
    for loop in G._node:
        G._node[loop]['index'] = ind
        G._node[loop]['link'] = list(nx.neighbors(G,loop))
        G._node[loop]['numlink'] = len(list(nx.neighbors(G,loop)))
        lnk[ind] = len(list(nx.neighbors(G,loop)))
        G._node[loop]['coupling'] = np.zeros(shape=(lnk[ind],))
        for linkloop in range (lnk[ind]):
            G._node[loop]['coupling'][linkloop] = g + (random.random() - random.random()) * param_perturb
        G._node[loop]['mu'] = mu + (random.random() - random.random()) * param_perturb
        G._node[loop]['alpha'] = alpha + (random.random() - random.random()) * param_perturb
        G._node[loop]['beta'] = beta + (random.random() - random.random()) * param_perturb
        G._node[loop]['c'] = c + (random.random() - random.random()) * param_perturb
        G._node[loop]['l'] = l + (random.random() - random.random()) * param_perturb
        G._node[loop]['k'] = k + (random.random() - random.random()) * param_perturb
        G._node[loop]['v'] = v + (random.random() - random.random()) * param_perturb
        G._node[loop]['m'] = m + (random.random() - random.random()) * param_perturb
        
        u0[ind*2 + 0] = random.random()
        u0[ind*2 + 1] = random.random()
        u0[ind*2 + 2] = random.random()
        # can add additional factor of 10 if we are using ISLM_model
        if ode_func == van_der_pol:
            u0[ind*2 + 0]*=5
            u0[ind*2 + 1]*=5
            u0[ind*2 + 2]*=5
        if ode_func == ISLM_model:
            u0[ind*2 + 0]*=50
            u0[ind*2 + 1]*=50
            u0[ind*2 + 2]*=50
        if ode_func == chaos_model:
            u0[ind*2 + 0]*=2
            u0[ind*2 + 1]*=2
            u0[ind*2 + 2]*=2
        

        # set the master initial conditions so we know what they are
        if G._node[loop]['name'] == 'Master':
            G._node[loop]['mu'] = mu
            G._node[loop]['alpha'] = alpha
            G._node[loop]['beta'] = beta
            G._node[loop]['c'] = c
            G._node[loop]['l'] = l
            G._node[loop]['k'] = k
            G._node[loop]['v'] = v
            G._node[loop]['m'] = m
            u0[ind*2 + 0] = x0
            u0[ind*2 + 1] = v0
            u0[ind*2 + 1] = y0
            for linkloop in range (lnk[ind]):
                G._node[loop]['coupling'][linkloop] = g
        ind+=1

    # set the time interval with time steps
    t = np.linspace(0, total_time, t_divisions)

    # run the ODE solver
    U = ode_func(G, u0, t)

    return U, t