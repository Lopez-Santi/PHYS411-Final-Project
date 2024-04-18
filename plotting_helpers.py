import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import networkx as nx
import time
import numpy.linalg as lin
import random
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

def vpd_plt(G, U, t):
    """
    Function will plot the position v.s. time, velocity v.s. time, and phase (velocity v.s. position)
    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just plot all nodes same color
        U : The solution from ODE solver
        t : The array time stepped interval
    """

    # inital arrays and parameters
    N = int(np.shape(U)[1] / 2)
    x = np.zeros(shape = (np.shape(U)[0], N))
    v = np.zeros(shape = (np.shape(U)[0], N))

    # Separating position and velocity into separate arrays
    for i in range(len(U)):
        ind = 0
        for j in range(N):
            x[i][j] = U[i][ind]
            v[i][j] = U[i][ind+1]
            ind+=2        

    # plotting position v.s. time
    plt.figure(figsize=(10,6))

    # satellite economies first
    c = 'cornflowerblue'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] != 'Master':
            p1, = plt.plot(t, x[:,ind], color = c)
        ind+=1
    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p2, = plt.plot(t, x[:,ind], color = c)
        ind+=1

    plt.title('Position Space')
    plt.xlabel('Time')
    plt.ylabel('Postion')
    plt.xlim(0, t[-1])
    plt.legend([p1, p2], ['Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

    # plotting velocity v.s. time
    plt.figure(figsize=(10,6))

    # satellite economies first
    c = 'cornflowerblue'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] != 'Master':
            p1, = plt.plot(t, v[:,ind], color = c)
        ind+=1
    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p2, = plt.plot(t, v[:,ind], color = c)
        ind+=1

    plt.title('Velocity Space')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.xlim(0, t[-1])
    plt.legend([p1, p2], ['Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

    # plotting phase portrait
    plt.figure(figsize=(10,6))
    
    # satellite economies first
    c = 'cornflowerblue'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] != 'Master':
            p1, = plt.plot(x[:,ind], v[:,ind], color = c)
        ind+=1
    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p2, = plt.plot(x[:,ind], v[:,ind], color = c)
        ind+=1

    plt.title('Phase Space')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend([p1, p2], ['Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()


def vpd_plt_deluxe(G, U, t):
    """
    Function will plot the position v.s. time, velocity v.s. time, and phase (velocity v.s. position)
    Also changes the color of the nodes directly attatched to master and adds labels
    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just plot all nodes same color
        U : The solution from ODE solver
        t : The array time stepped interval
    """

    # inital arrays and parameters
    N = int(np.shape(U)[1] / 2)
    x = np.zeros(shape = (np.shape(U)[0], N))
    v = np.zeros(shape = (np.shape(U)[0], N))

    # Separating position and velocity into separate arrays
    for i in range(len(U)):
        ind = 0
        for j in range(N):
            x[i][j] = U[i][ind]
            v[i][j] = U[i][ind+1]
            ind+=2        

    # plotting position v.s. time
    plt.figure(figsize=(10,6))

    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(t, x[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(t, x[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(t, x[:,ind], color = c)
        ind+=1

    plt.title('Position Space')
    plt.xlabel('Time')
    plt.ylabel('Postion')
    plt.xlim(0, t[-1])
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

    # plotting velocity v.s. time
    plt.figure(figsize=(10,6))

   # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(t, v[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(t, v[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(t, v[:,ind], color = c)
        ind+=1

    plt.title('Velocity Space')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.xlim(0, t[-1])
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

    # plotting phase portrait
    plt.figure(figsize=(10,6))
    
    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(x[:,indx], v[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(x[:,indx], v[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(x[:,ind], v[:,ind], color = c)
        ind+=1

    plt.title('Phase Space')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()


def ISLM_plt(G, U, t):
    """
    Function will plot the Interest Rate v.s. time, Real Income v.s. time, and phase (Interest Rate v.s. Real Income)
    Also changes the color of the nodes directly attatched to master and adds labels
    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just plot all nodes same color
        U : The solution from ODE solver
        t : The array time stepped interval
    """

    # inital arrays and parameters
    N = int(np.shape(U)[1] / 2)
    r = np.zeros(shape = (np.shape(U)[0], N))
    g = np.zeros(shape = (np.shape(U)[0], N))

    # Separating interest rate and real income into separate arrays
    for i in range(len(U)):
        ind = 0
        for j in range(N):
            r[i][j] = U[i][ind]
            g[i][j] = U[i][ind+1]
            ind+=2        

    # plotting Interest Rate v.s. time
    plt.figure(figsize=(10,6))

    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(t, r[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(t, r[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(t, r[:,ind], color = c)
        ind+=1

    plt.title('Interest Rate Space')
    plt.xlabel('Time')
    plt.ylabel('Interest Rate')
    plt.xlim(0, t[-1])
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

    # plotting Real Income v.s. time
    plt.figure(figsize=(10,6))

   # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(t, g[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(t, g[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(t, g[:,ind], color = c)
        ind+=1

    plt.title('Real Income Space')
    plt.xlabel('Time')
    plt.ylabel('Real Income')
    plt.xlim(0, t[-1])
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

    # plotting phase portrait
    plt.figure(figsize=(10,6))
    
    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(g[:,indx], r[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(g[:,indx], r[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(g[:,ind], r[:,ind], color = c)
        ind+=1

    plt.title('Phase Space')
    plt.xlabel('Real Income')
    plt.ylabel('Interest Rate')
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

def kaldor_plt(G, U, t):
    """
    Function will plot the Capital Stock v.s. time, Output v.s. time, and phase (Capital Stock v.s. Output)
    Also changes the color of the nodes directly attatched to master and adds labels
    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just plot all nodes same color
        U : The solution from ODE solver
        t : The array time stepped interval
    """

    # inital arrays and parameters
    N = int(np.shape(U)[1] / 2)
    r = np.zeros(shape = (np.shape(U)[0], N))
    g = np.zeros(shape = (np.shape(U)[0], N))

    # Separating Capital Stock and output into separate arrays
    for i in range(len(U)):
        ind = 0
        for j in range(N):
            r[i][j] = U[i][ind]
            g[i][j] = U[i][ind+1]
            ind+=2        

    # plotting capital stock v.s. time
    plt.figure(figsize=(10,6))

    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(t, r[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(t, r[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(t, r[:,ind], color = c)
        ind+=1

    plt.title('Capital Stock Space')
    plt.xlabel('Time')
    plt.ylabel('Capital Stock')
    plt.xlim(0, t[-1])
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

    # plotting output v.s. time
    plt.figure(figsize=(10,6))

   # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(t, g[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(t, g[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(t, g[:,ind], color = c)
        ind+=1

    plt.title('Output Space')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.xlim(0, t[-1])
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()

    # plotting phase portrait
    plt.figure(figsize=(10,6))
    
    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = plt.plot(g[:,indx], r[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = plt.plot(g[:,indx], r[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = plt.plot(g[:,ind], r[:,ind], color = c)
        ind+=1

    plt.title('Phase Space')
    plt.xlabel('Output')
    plt.ylabel('Capital Stock')
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()


def chaos_plt(G, U, t):
    """
    Function will plot the
    - Individually : inventory Stock v.s. time, Output v.s. time, sales v.s. time
    - Phase : inventory Stock v.s. Output, sales v.s. inventory stock, sales v.s. Output
    - 3-D : inventory Stock v.s. Output v.s. sales
    Also changes the color of the nodes directly attatched to master and adds labels
    Arguments:
        G : The graph (networkx) with node having the name 'Master'
            If there is no node named 'Master' then the function will just plot all nodes same color
        U : The solution from ODE solver
        t : The array time stepped interval
    """

    # inital arrays and parameters
    N = int(np.shape(U)[1] / 3)
    x = np.zeros(shape = (np.shape(U)[0], N)) # sales
    v = np.zeros(shape = (np.shape(U)[0], N)) # output
    y = np.zeros(shape = (np.shape(U)[0], N)) # inventory stock

    # Separating interest rate and real income into separate arrays
    for i in range(len(U)):
        ind = 0
        for j in range(N):
            x[i][j] = U[i][ind]
            v[i][j] = U[i][ind+1]
            y[i][j] = U[i][ind+2]
            ind+=2        


    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(10,6)

    ### INDIVIDUAL PLOTS ###

    # plotting sales v.s. time
    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = axs[0, 0].plot(t, x[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = axs[0, 0].plot(t, x[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = axs[0, 0].plot(t, x[:,ind], color = c)
        ind+=1

    axs[0, 0].set_title('Sales Space')
    axs[0, 0].set(xlabel='Time',ylabel='Sales')

    # plotting output v.s. time
   # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = axs[0, 1].plot(t, v[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = axs[0, 1].plot(t, v[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = axs[0, 1].plot(t, v[:,ind], color = c)
        ind+=1

    axs[0, 1].set_title('Output Space')
    axs[0, 1].set(xlabel='Time',ylabel='Output')


    # plotting inventory stock v.s. time
   # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = axs[0, 2].plot(t, y[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = axs[0, 2].plot(t, y[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = axs[0, 2].plot(t, y[:,ind], color = c)
        ind+=1

    axs[0, 2].set_title('Inventory Stock Space')
    axs[0, 2].set(xlabel='Time',ylabel='Inventory Stock')
    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})

    # ### PHASE PORTRAITS ### sales (x), output (v), inventory stocks (y)
    #- Phase : inventory Stock v.s. Output, sales v.s. inventory stock, sales v.s. Output
    # plotting inventory Stock v.s. Output
    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = axs[1, 0].plot(v[:,indx], y[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = axs[1, 0].plot(v[:,indx], y[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = axs[1, 0].plot(v[:,ind], y[:,ind], color = c)
        ind+=1

    axs[1, 0].set_title('Inventory Stock v.s. Output')
    axs[1, 0].set(xlabel='Output',ylabel='Inventory Stock')

    # plotting sales v.s. inventory stock
   # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = axs[1, 1].plot(y[:,indx], x[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = axs[1, 1].plot(y[:,indx], x[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = axs[1, 1].plot(y[:,ind], x[:,ind], color = c)
        ind+=1

    axs[1, 1].set_title('Sales v.s. Inventory Stock')
    axs[1, 1].set(xlabel='Inventory Stock',ylabel='Sales')


    # plotting sales v.s. Output
   # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = axs[1, 2].plot(x[:,indx], v[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = axs[1, 2].plot(x[:,indx], v[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = axs[1, 2].plot(x[:,ind], v[:,ind], color = c)
        ind+=1

    axs[1, 2].set_title('Output v.s. Sales')
    axs[1, 2].set(xlabel='Sales',ylabel='Output')
    # plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
    #            handler_map={tuple: HandlerTuple(ndivide=None)})
    
    fig.tight_layout()

    ### PLOTTING 3-D ###
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    """                                                                                                                                                    
    Scaling is done from here...                                                                                                                           
    """
    x_scale=1
    y_scale=1
    z_scale=1

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj=short_proj
    """                                                                                                                                                    
    to here                                                                                                                                                
    """
    # satellite economies first #
    ind = 0
    # iterate over each node
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master
        if G._node[i]['name'] != 'Master':
            for gloop in range(linksz):
                c = 'cornflowerblue'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                # add the lines for all nodes not connected to master and that have not been added already
                if G._node[cindex]['name'] != 'Master' and indx > ind:
                    p1, = ax.plot(x[:,indx], v[:,indx], y[:,indx], color = c)
        ind += 1
    ind = 0
    # now plotting the satellites directly connected to master
    for i in G._node:
        linksz = G._node[i]['numlink'] # check the number of links each node has
        # add coupling only if we are not the master (master evolves normally)       
        if G._node[i]['name'] == 'Master':
            for gloop in range(linksz):
                c = 'green'
                cindex = G._node[i]['link'][gloop]
                indx = G._node[cindex]['index']
                p2, = ax.plot(x[:,indx], v[:,indx], y[:,indx], color = c) # add the lines for all nodes directly connected to master
        ind += 1

    # now the master economy
    c = 'red'
    ind = 0
    for node in G._node:
        if G._node[node]['name'] == 'Master':
            p3, = ax.plot(x[:,ind], v[:,ind], y[:,ind], color = c)
        ind+=1

    ax.set_xlabel("Sales")
    ax.set_ylabel("Output")
    ax.set_zlabel("Inventory Stock")
    ax.set_title('Sales v.s. Output v.s. Inventory Stock')

    plt.legend([p1, p2, p3], ['Indirectly Connected Satellite', 'Directly Connected Satellite', 'Master'],
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.show()
