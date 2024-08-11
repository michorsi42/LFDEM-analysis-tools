import os
import re
import sys
import glob
import shutil
import colorsys
import matplotlib
import rigidClusterProcessor
import numpy             as     np
import matplotlib.pyplot as     plt
from   matplotlib        import colors
from   collections       import defaultdict





#%% MERGE COMMON

# function to merge sublists having common elements
# link: https://www.geeksforgeeks.org/python-merge-list-with-common-elements-in-a-list-of-lists/
def merge_common(lists):
    neigh = defaultdict(set)
    visited = set()
    for each in lists:
        for item in each:
            neigh[item].update(each)
    def comp(node, neigh = neigh, visited = visited, vis = visited.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            vis(node)
            nodes |= neigh[node] - visited
            yield node
    for node in neigh:
        if node not in visited:
            yield sorted(comp(node))





#%% RIGID CLUSTERS (PEBBLE GAME)

def myRigidClusters(Dir):
    
    rigidClusterProcessor.rigFileGenerator(Dir,Dir)
    
    # Read in rig_ files and write n_rigid.csv files
    rigFile = Dir + 'rig_' + os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    (rigidClusterSizes,numBonds,clusterIDs) = rigidClusterProcessor.rigFileReader(rigFile)
    
    n_rigid = np.array([])
    for clusterLists in rigidClusterSizes:
        n_rigid = np.append(n_rigid, np.sum(np.array(clusterLists)))
    n_rigid = np.array([int(x) for x in n_rigid])
    
    np.savetxt(Dir+'F_rig.txt', np.transpose([n_rigid]), delimiter=' ', fmt='%f')





#%% PRIME RIGID CLUSTERS

def myPrimeRigidClusters(Dir):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName
    intFile  = Dir + 'int_'  + baseName
    
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
    
    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData   = []
    counter   = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime  = True
                counter   += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")
    del file, fileLines, line, counter, isNewTime
    
    with open(rigFile) as file:
        fileLines = file.readlines()[1:ndt+1]
    clustersSizes = []
    for line in fileLines:
        clustersSizes.append([int(item) for item in line.split()])
    del file, fileLines, line
    
    with open(rigFile) as file:
        fileLines = file.readlines()[2*ndt+5:]
    rigPartsIDs = []
    counter     = -1
    isNewTime   = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                rigPartsIDs.append([])
                isNewTime  = True
                counter   += 1
        else:
            IDs = np.unique([int(item) for item in line.split(",")])
            if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
            rigPartsIDs[counter].append(IDs)
            isNewTime = False
    if len(rigPartsIDs) != ndt:
        sys.exit("ERROR: something's wrong with the reading of rigFile")
    del file, fileLines, line, counter, isNewTime, IDs
    
    F_prime_rig        = np.zeros(ndt, dtype=int)
    primeClusters      = []
    primeClustersSizes = []
    
    for it in range(ndt):
        
        ip, jp, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, contState, dummy, dummy, dummy, dummy, dummy, dummy = np.reshape(intData[it], (len(intData[it]), 17)).T
        
        ip            = np.array(ip,        dtype=int)
        jp            = np.array(jp,        dtype=int)
        contState     = np.array(contState, dtype=int)
        frictContacts = np.where(contState>1)[0]
        
        rigPartsIDs_it = rigPartsIDs[it]
        if len(rigPartsIDs_it) > 0:
            rigPartsIDs_it = np.concatenate(rigPartsIDs_it)
        
        partsInCont = [[] for i in range(NP)]
        for i in frictContacts:
            partsInCont[ip[i]].append(jp[i])
            partsInCont[jp[i]].append(ip[i])

        rigPrimePartsIDs = []
        for i in range(NP):
            if i in rigPartsIDs_it:
                isItPrime = True
                for j in partsInCont[i]:
                    if j not in rigPartsIDs_it:
                        isItPrime = False
                        break
                if isItPrime:
                    rigPrimePartsIDs.append(i)
                    
        F_prime_rig[it] = len(rigPrimePartsIDs)
        
        primeClusters_it = []
        for i in frictContacts:
            if ip[i] in rigPrimePartsIDs and jp[i] not in rigPrimePartsIDs:
                ip_already_in_prime_cluster = False
                for j in range(len(primeClusters_it)):
                    if ip[i] in primeClusters_it[j]:
                        ip_already_in_prime_cluster = True
                        break
                if not ip_already_in_prime_cluster:
                    primeClusters_it.append([ip[i]])
            elif ip[i] not in rigPrimePartsIDs and jp[i] in rigPrimePartsIDs:
                jp_already_in_prime_cluster = False
                for j in range(len(primeClusters_it)):
                    if jp[i] in primeClusters_it[j]:
                        jp_already_in_prime_cluster = True
                        break
                if not jp_already_in_prime_cluster:
                    primeClusters_it.append([jp[i]])
            elif ip[i] in rigPrimePartsIDs and jp[i] in rigPrimePartsIDs:
                ip_already_in_prime_cluster = False
                jp_already_in_prime_cluster = False
                for j in range(len(primeClusters_it)):
                    if ip[i] in primeClusters_it[j] and jp[i] not in primeClusters_it[j]:
                        ip_already_in_prime_cluster = True
                        jp_already_in_prime_cluster = True
                        primeClusters_it[j].append(jp[i])
                    elif ip[i] not in primeClusters_it[j] and jp[i] in primeClusters_it[j]:
                        ip_already_in_prime_cluster = True
                        jp_already_in_prime_cluster = True
                        primeClusters_it[j].append(ip[i])
                if not ip_already_in_prime_cluster and not jp_already_in_prime_cluster:
                    primeClusters_it.append([ip[i], jp[i]])
                    
        primeClusters_it = list(merge_common(primeClusters_it))

        if len(primeClusters_it) > 0:
            if len(np.concatenate(primeClusters_it)) != len(rigPrimePartsIDs):
                sys.exit("ERROR: something's wrong with clusters_it")
        elif len(primeClusters_it) == 0 and len(rigPrimePartsIDs) != 0:
            sys.exit("ERROR: something's wrong with clusters_it")

        primeClusters.append(primeClusters_it)
        primeClustersSizes.append([len(primeClusters_it[i]) for i in range(len(primeClusters_it))])
        
    FPrimeFile = open(Dir+"F_prime_rig.txt", "w")
    FPrimeFile.write("t                F'_rig" + '\n')
    for it in range(ndt):
        FPrimeFile.write('{:.4f}'.format(t[it]) + '      ' + str(F_prime_rig[it])  + '\n')
    FPrimeFile.close()
    
    rigPrimeFile = open(Dir+"rigPrime.txt", "w")
    rigPrimeFile.write("#Prime Rigid Clusters Sizes" + '\n')
    for it in range(ndt):
        if len(primeClustersSizes[it]) == 0:
            rigPrimeFile.write("0   \n")
        else:
            for i in range(len(primeClustersSizes[it])):
                rigPrimeFile.write(str(primeClustersSizes[it][i]) + '   ')
            rigPrimeFile.write("\n")
    rigPrimeFile.write("\n")
    rigPrimeFile.write("#Prime Rigid Clusters IDs" + '\n')
    for it in range(ndt):
        rigPrimeFile.write('#snapshot = ' + str(it) + '\n')
        if len(primeClustersSizes[it]) == 0:
            rigPrimeFile.write("0\n")
        else:
            for i in range(len(primeClusters[it])):
                for j in range(len(primeClusters[it][i])):
                    if j < len(primeClusters[it][i])-1:
                        rigPrimeFile.write(str(primeClusters[it][i][j]) + ',')
                    else:
                        rigPrimeFile.write(str(primeClusters[it][i][j]))
                rigPrimeFile.write("\n")
    rigPrimeFile.close()





#%% CLUSTER SIZE DISTRIBUTION

def cluster_size_distribution(Dir, SSi):
    
    baseName     = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile     = Dir + 'data_' + baseName
    rigFile      = Dir + 'rig_'  + baseName
    rigPrimeFile = Dir + 'rigPrime.txt'
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
    
    with open(rigFile) as file:
        fileLines = file.readlines()[1:ndt+1]
    clustersSizes = []
    for line in fileLines:
        clustersSizes.append([int(item) for item in line.split()])
    del file, fileLines, line
    
    with open(rigPrimeFile) as file:
        fileLines = file.readlines()[1:ndt+1]
    clustersSizes_prime = []
    for line in fileLines:
        clustersSizes_prime.append([int(item) for item in line.split()])
    del file, fileLines, line
        
    clustersSizes       = [item for sublist in clustersSizes       for item in sublist]
    clustersSizes_prime = [item for sublist in clustersSizes_prime for item in sublist]
    
    clustersSizes       = [i for i in clustersSizes       if i!=0]
    clustersSizes_prime = [i for i in clustersSizes_prime if i!=0]
    
    if clustersSizes       == []:   clustersSizes       = [0]
    if clustersSizes_prime == []:   clustersSizes_prime = [0]
    
    cedges          = np.arange(np.min(clustersSizes)-0.5,       np.max(clustersSizes)+1.5)
    cbins           = np.arange(np.min(clustersSizes),           np.max(clustersSizes)+1)
    
    cedges_prime    = np.arange(np.min(clustersSizes_prime)-0.5, np.max(clustersSizes_prime)+1.5)
    cbins_prime     = np.arange(np.min(clustersSizes_prime),     np.max(clustersSizes_prime)+1)
    
    Pn,       dummy = np.histogram(clustersSizes[SSi:],       cedges,       density=True)
    Pn_prime, dummy = np.histogram(clustersSizes_prime[SSi:], cedges_prime, density=True)
    
    Pn_file      = open(Dir+"Pn.txt", "w")
    Pn_even_file = open(Dir+"Pn_even.txt", "w")
    Pn_odd_file  = open(Dir+"Pn_odd.txt", "w")
    Pn_file.write('n      P(n)' + '\n')
    Pn_even_file.write('n      P(n)' + '\n')
    Pn_odd_file.write('n      P(n)' + '\n')
    for i_n in range(len(cbins)):
        Pn_file.write('{:.0f}'.format(cbins[i_n]) + '      ' + '{:.9f}'.format(Pn[i_n]) + '\n')
        if cbins[i_n]%2==0:   # even
            Pn_even_file.write('{:.0f}'.format(cbins[i_n]) + '      ' + '{:.9f}'.format(Pn[i_n]) + '\n')
        else:   # odd
            Pn_odd_file.write('{:.0f}'.format(cbins[i_n]) + '      ' + '{:.9f}'.format(Pn[i_n]) + '\n')
    Pn_file.close()
    Pn_even_file.close()
    Pn_odd_file.close()
    
    Pn_prime_file      = open(Dir+"Pn_prime.txt", "w")
    Pn_prime_even_file = open(Dir+"Pn_prime_even.txt", "w")
    Pn_prime_odd_file  = open(Dir+"Pn_prime_odd.txt", "w")
    Pn_prime_file.write("n'      P(n')" + '\n')
    Pn_prime_even_file.write("n'      P(n')" + '\n')
    Pn_prime_odd_file.write("n'      P(n')" + '\n')
    for i_n in range(len(cbins_prime)):
        Pn_prime_file.write('{:.0f}'.format(cbins_prime[i_n]) + '      ' + '{:.9f}'.format(Pn_prime[i_n]) + '\n')
        if cbins_prime[i_n]%2==0:   # even
            Pn_prime_even_file.write('{:.0f}'.format(cbins_prime[i_n]) + '      ' + '{:.9f}'.format(Pn_prime[i_n]) + '\n')
        else:   # odd
            Pn_prime_odd_file.write('{:.0f}'.format(cbins_prime[i_n]) + '      ' + '{:.9f}'.format(Pn_prime[i_n]) + '\n')
    Pn_prime_file.close()
    Pn_prime_even_file.close()
    Pn_prime_odd_file.close()





#%% C>=n PARTICLES

def C_parts_IDs(Dir):

    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName
    
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
        
    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData   = []
    counter   = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime  = True
                counter   += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")
    del file, fileLines, line, counter, isNewTime
    
    C3PartsIDs = []
    C6PartsIDs = []
    
    for it in range(ndt):
        
        ip, jp, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, contState, dummy, dummy, dummy, dummy, dummy, dummy = np.reshape(intData[it], (len(intData[it]), 17)).T
        
        ip            = np.array(ip,        dtype=int)
        jp            = np.array(jp,        dtype=int)
        contState     = np.array(contState, dtype=int)
        stickContacts = np.where(contState==2)[0]
        slideContacts = np.where(contState==3)[0]
        
        numConstPerPart = np.zeros(NP, dtype=int)
        for i in range(stickContacts.size):
            numConstPerPart[ip[stickContacts[i]]] += 2
            numConstPerPart[jp[stickContacts[i]]] += 2
        for i in range(slideContacts.size):
            numConstPerPart[ip[slideContacts[i]]] += 1
            numConstPerPart[jp[slideContacts[i]]] += 1
        
        C3PartsIDs.append(np.where(numConstPerPart>=3)[0])
        C6PartsIDs.append(np.where(numConstPerPart>=6)[0])
        
    C3PartsIDsFile = open(Dir+"C3PartsIDs.txt", "w")
    for it in range(ndt):
        if len(C3PartsIDs[it]) == 0:
            C3PartsIDsFile.write("none")
        else:
            for ip in C3PartsIDs[it]:
                C3PartsIDsFile.write(str(ip) + "   ")
        C3PartsIDsFile.write('\n')
    C3PartsIDsFile.close()
    
    C6PartsIDsFile = open(Dir+"C6PartsIDs.txt", "w")
    for it in range(ndt):
        if len(C6PartsIDs[it]) == 0:
            C6PartsIDsFile.write("none")
        else:
            for ip in C6PartsIDs[it]:
                C6PartsIDsFile.write(str(ip) + "   ")
        C6PartsIDsFile.write('\n')
    C6PartsIDsFile.close()





#%% Z & C

def Z_C(Dir):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName
    
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
    
    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData   = []
    counter   = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime  = True
                counter   += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")
    del file, fileLines, line, counter, isNewTime
        
    Z_Znet = np.zeros((ndt,4))
    C_Cnet = np.zeros((ndt,4))
    
    for it in range(ndt):
        
        ip, jp, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, contState, dummy, dummy, dummy, dummy, dummy, dummy = np.reshape(intData[it], (len(intData[it]), 17)).T
                    
        ip            = np.array(ip,        dtype=int)
        jp            = np.array(jp,        dtype=int)
        contState     = np.array(contState, dtype=int)
        frictContacts = np.where(contState> 1)[0]
        stickContacts = np.where(contState==2)[0]
        slideContacts = np.where(contState==3)[0]
        
        ConstraintsPerPart   = np.zeros(NP, dtype=int)
        numContsPerPart      = np.zeros(NP, dtype=int)
        numStickContsPerPart = np.zeros(NP, dtype=int)
        
        for i in range(frictContacts.size):
            numContsPerPart[ip[frictContacts[i]]] += 1
            numContsPerPart[jp[frictContacts[i]]] += 1
        
        for i in range(stickContacts.size):
            ConstraintsPerPart[ip[stickContacts[i]]]   += 2
            ConstraintsPerPart[jp[stickContacts[i]]]   += 2
            numStickContsPerPart[ip[stickContacts[i]]] += 1
            numStickContsPerPart[jp[stickContacts[i]]] += 1
        
        for i in range(slideContacts.size):
            ConstraintsPerPart[ip[slideContacts[i]]] += 1
            ConstraintsPerPart[jp[slideContacts[i]]] += 1
        
        if stickContacts.size > 0:
            Z_Znet[it][0] = np.mean(numStickContsPerPart)
            Z_Znet[it][1] = np.std(numStickContsPerPart)
            Z_Znet[it][2] = np.mean(numStickContsPerPart[numStickContsPerPart!=0])
            Z_Znet[it][3] = np.std(numStickContsPerPart[numStickContsPerPart!=0])
        
        if frictContacts.size > 0:
            C_Cnet[it][0] = np.mean(ConstraintsPerPart)
            C_Cnet[it][1] = np.std(ConstraintsPerPart)
            C_Cnet[it][2] = np.mean(ConstraintsPerPart[numContsPerPart!=0])
            C_Cnet[it][3] = np.std(ConstraintsPerPart[numContsPerPart!=0])
    
    np.savetxt(Dir+"Z_Znet.txt", Z_Znet, delimiter='      ', fmt='%.9f', header='mean(Z)      std(Z)      mean(Znet)      std(Znet)')
    np.savetxt(Dir+"C_Cnet.txt", C_Cnet, delimiter='      ', fmt='%.9f', header='mean(C)      std(C)      mean(Cnet)      std(Cnet)')





#%% RIGIDITY PERSISTENCE

def rigPers(Dir, SSi, outputVar):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName
    
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)
        
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
    
    ndt = len(t)
        
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    with open(rigFile) as file:
        fileLines = file.readlines()[2*ndt+5:]
    rigPartsIDs = []
    counter     = -1
    isNewTime   = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                rigPartsIDs.append([])
                isNewTime  = True
                counter   += 1
        else:
            IDs = np.unique([int(item) for item in line.split(",")])
            if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
            rigPartsIDs[counter].append(IDs)
            isNewTime = False
    if len(rigPartsIDs) != ndt:
        sys.exit("ERROR: something's wrong with the reading of rigFile")
    del file, fileLines, line, counter, isNewTime, IDs
    
    isInCluster = np.zeros((ndt-SSi,NP), dtype=bool)
    
    for it in range(SSi,ndt):
        rigPartsIDs_it = rigPartsIDs[it]
        if len(rigPartsIDs_it) > 0:
            rigPartsIDs_it = np.concatenate(rigPartsIDs_it)
        for ip in rigPartsIDs_it:
            isInCluster[it-SSi][ip] = True
                
    ntaus      = ndt-SSi
    rigPers    = np.zeros(ntaus)
    corrProd   = np.zeros((ntaus,NP))
    uncorrProd = np.zeros((ntaus,NP))
    for it1 in range(ntaus):
        uncorrProd[it1] += isInCluster[it1]
        for it2 in range(it1,ntaus):
            k            = it2 - it1
            corrProd[k] += isInCluster[it1] * isInCluster[it2]
    for k in range(ntaus):
        corrProd[k]   /= (ntaus-k)
        uncorrProd[k] /= (ntaus-k)
        rigPers[k]     = np.sum(corrProd[k] - uncorrProd[k]**2.)
        
    if outputVar == 't':
        delta   = t[1]-t[0]
        header  = 'Delta t       C'
    elif outputVar == 'gamma':
        delta   = gamma[1]-gamma[0]
        header  = 'Delta gamma       C'
    else:
        sys.exit("ERROR: there is a problem with outputVar")
        
    rigPersFile = open(Dir+"rigPers.txt", "w")
    rigPersFile.write(header + '\n')
    for k in range(ntaus):
        rigPersFile.write(str(round(delta*k,9)) + '      ' +
                          str(rigPers[k])       + '\n')
    rigPersFile.close()





#%% NUMBER OF CONSTRAINTS PERSISTENCE

def constPers(Dir, SSi, outputVar):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName
    
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
    
    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData   = []
    counter   = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime  = True
                counter   += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")
    del file, fileLines, line, counter, isNewTime
    
    isItC3 = np.zeros((ndt-SSi,NP), dtype=bool)
    isItC6 = np.zeros((ndt-SSi,NP), dtype=bool)
    
    for it in range(ndt):
        if it >= SSi:
            ip, jp, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, contState, dummy, dummy, dummy, dummy, dummy, dummy = np.reshape(intData[it], (len(intData[it]), 17)).T
            ip              = np.array(ip,        dtype=int)
            jp              = np.array(jp,        dtype=int)
            contState       = np.array(contState, dtype=int)
            stickContacts   = np.where(contState==2)[0]
            slideContacts   = np.where(contState==3)[0]
            numConstPerPart = np.zeros(NP, dtype=int)
            for i in range(stickContacts.size):
                numConstPerPart[ip[stickContacts[i]]] += 2
                numConstPerPart[jp[stickContacts[i]]] += 2
            for i in range(slideContacts.size):
                numConstPerPart[ip[slideContacts[i]]] += 1
                numConstPerPart[jp[slideContacts[i]]] += 1
            isItC3[it-SSi] = numConstPerPart >= 2
            isItC6[it-SSi] = numConstPerPart >= 3
    
    ntaus        = ndt-SSi
    contPersC3   = np.zeros(ntaus)
    contPersC6   = np.zeros(ntaus)
    corrProdC3   = np.zeros((ntaus,NP))
    corrProdC6   = np.zeros((ntaus,NP))
    uncorrProdC3 = np.zeros((ntaus,NP))
    uncorrProdC6 = np.zeros((ntaus,NP))
    for it1 in range(ntaus):
        uncorrProdC3[it1] += isItC3[it1]
        uncorrProdC6[it1] += isItC6[it1]
        for it2 in range(it1,ntaus):
            k              = it2 - it1
            corrProdC3[k] += isItC3[it1] * isItC3[it2]
            corrProdC6[k] += isItC6[it1] * isItC6[it2]
    for k in range(ntaus):
        corrProdC3[k]   /= (ntaus-k)
        corrProdC6[k]   /= (ntaus-k)
        uncorrProdC3[k] /= (ntaus-k)
        uncorrProdC6[k] /= (ntaus-k)
        contPersC3[k]    = np.sum(corrProdC3[k] - uncorrProdC3[k]**2.)
        contPersC6[k]    = np.sum(corrProdC6[k] - uncorrProdC6[k]**2.)
    
    if outputVar == 't':
        delta   = t[1]-t[0]
        header  = 'Delta t       C'
    elif outputVar == 'gamma':
        delta   = gamma[1]-gamma[0]
        header  = 'Delta gamma       C'
    else:
        sys.exit("ERROR: there is a problem with outputVar")
        
    contPersC3File = open(Dir+"contC3Pers.txt", "w")
    contPersC3File.write(header + '\n')
    for k in range(ntaus):
        contPersC3File.write(str(round(delta*k,9)) + '      ' +
                             str(contPersC3[k])    + '\n')
    contPersC3File.close()
    
    contPersC6File = open(Dir+"contC6Pers.txt", "w")
    contPersC6File.write(header + '\n')
    for k in range(ntaus):
        contPersC6File.write(str(round(delta*k,9)) + '      ' +
                             str(contPersC6[k])    + '\n')
    contPersC6File.close()





#%% MAX CLUSTER SIZE

def maxClusterSize(Dir):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName
    
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)
        
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
        
    maxClustersSize = [np.max([int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=it+1, max_rows=1))]) for it in range(ndt)]
        
    nMaxFile = open(Dir+"maxClusterSize.txt", "w")
    for k in range(ndt):
        nMaxFile.write(str(maxClustersSize[k]) + '\n')
    nMaxFile.close()





#%% MAX CLUSTER SIZE TIME CORRELATION FUNCTION

def maxClusterSize_corr(Dir, SSi, outputVar):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    nMaxFile = Dir + 'maxClusterSize.txt'
    
    # this function requires maxClusterSize to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(nMaxFile):
        maxClusterSize(Dir, SSi)
        
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
        
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    maxClustersSize = np.loadtxt(nMaxFile).transpose()
    maxClustersSize = maxClustersSize[SSi:]
        
    ntaus       = ndt-SSi
    nMaxPers    = np.zeros(ntaus)
    corrProd    = np.zeros((ntaus,NP))
    uncorrProd1 = np.zeros((ntaus,NP))
    uncorrProd2 = np.zeros((ntaus,NP))
    for it1 in range(ntaus):
        for it2 in range(it1,ntaus):
            k               = it2 - it1
            corrProd[k]    += maxClustersSize[it1] * maxClustersSize[it2]
            uncorrProd1[k] += maxClustersSize[it1]
            uncorrProd2[k] += maxClustersSize[it2]
    for k in range(ntaus):
        corrProd[k]    /= (ntaus-k)
        uncorrProd1[k] /= (ntaus-k)
        uncorrProd2[k] /= (ntaus-k)
        nMaxPers[k]     = np.sum(corrProd[k] - uncorrProd1[k]*uncorrProd2[k])
        
    if outputVar == 't':
        delta   = t[1]-t[0]
        header  = 'Delta t       C'
    elif outputVar == 'gamma':
        delta   = gamma[1]-gamma[0]
        header  = 'Delta gamma       C'
    else:
        sys.exit("ERROR: there is a problem with outputVar")
        
    nMaxPersFile = open(Dir+"maxClusterSize_corr.txt", "w")
    nMaxPersFile.write(header + '\n')
    for k in range(ntaus):
        nMaxPersFile.write(str(round(delta*k,9)) + '      ' +
                           str(nMaxPers[k])      + '\n')
    nMaxPersFile.close()





#%% PAIR DISTRIBUTION FUNCTION

def PDF(Dir, SSi, dr, dtheta, partsTypePDF):
    
    if partsTypePDF != 'all' and partsTypePDF != 'rig' and partsTypePDF != 'rigPrime':
        sys.exit("ERROR: partsType can only be 'all', 'rig', or 'rigPrime'")
    
    baseName     = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile     = Dir + 'data_' + baseName
    parFile      = Dir + 'par_'  + baseName
    rigFile      = Dir + 'rig_'  + baseName
    rigPrimeFile = Dir + 'rigPrime.txt'
    
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])

    t,     gamma, gammapt, dummy,  dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy,   minGap, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy,   dummy,  dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    SSi = np.where(gamma[1:]>=1)[0][0]
        
    ndt = len(t)

    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]

    dummy, a, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
        
    dummy, dummy, rx, rz, vx, dummy, vz, dummy, omy, dummy, dummy = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)
    
    if partsTypePDF == 'all':
        partsIDs = np.arange(NP)
    elif partsTypePDF == 'rig':
        with open(rigFile) as file:
            fileLines = file.readlines()[2*ndt+5:]
        partsIDs  = []
        counter   = -1
        isNewTime = False
        for line in fileLines:
            if "#" in line:
                if not isNewTime:
                    partsIDs.append([])
                    isNewTime  = True
                    counter   += 1
            else:
                IDs = np.unique([int(item) for item in line.split(",")])
                if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
                partsIDs[counter].append(IDs)
                isNewTime = False
        if len(partsIDs) != ndt:
            sys.exit("ERROR: something's wrong with the reading of rigFile")
        del file, fileLines, line, counter, isNewTime, IDs
    elif partsTypePDF == 'rigPrime':
        with open(rigPrimeFile) as file:
            fileLines = file.readlines()[ndt+3:]
        partsIDs  = []
        counter   = -1
        isNewTime = False
        for line in fileLines:
            if "#" in line:
                if not isNewTime:
                    partsIDs.append([])
                    isNewTime  = True
                    counter   += 1
            else:
                IDs = np.unique([int(item) for item in line.split(",")])
                if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
                partsIDs[counter].append(IDs)
                isNewTime = False
        if len(partsIDs) != ndt:
            sys.exit("ERROR: something's wrong with the reading of rigPrimeFile")
        del file, fileLines, line, counter, isNewTime, IDs
    
    dtheta  *= np.pi / 180.
    rmin     = np.min(a) + np.min(minGap)   # to allow overlapping
    rmax     = np.max([Lx,Lz]) / 2.
    thetamin = -np.pi
    thetamax =  np.pi
    rlin     = np.arange(rmin,     rmax+dr,         dr)
    thetalin = np.arange(thetamin, thetamax+dtheta, dtheta)

    gr       = np.zeros( len(rlin)-1)
    grtheta  = np.zeros((len(rlin)-1, len(thetalin)-1))

    surf     = Lx * Lz
    prog_old = -1
    rho      = 0

    for it in range(SSi,ndt):
        
        if partsTypePDF == 'all':
            partsIDs_it = partsIDs
        else:
            partsIDs_it = partsIDs[it]
            if len(partsIDs_it) > 0:
                partsIDs_it = np.array([int(i) for i in np.concatenate(partsIDs_it)])
            else:
                partsIDs_it = []
            
        NPPDF = len(partsIDs_it)
        rho  += NPPDF / surf
        
        if NPPDF > 0:
            
            xp   = rx[it][partsIDs_it]
            zp   = rz[it][partsIDs_it]
            
            xmat = np.outer(xp, np.ones(NPPDF))
            dxij = xmat.transpose() - xmat
            
            zmat = np.outer(zp, np.ones(NPPDF))
            dzij = zmat.transpose() - zmat
            
            # z-periodity (Lees Edwards)
            dxij[dzij>Lz/2.]  -= gammapt[it] * Lz
            dzij[dzij>Lz/2.]  -= Lz
            dxij[dzij<-Lz/2.] += gammapt[it] * Lz
            dzij[dzij<-Lz/2.] += Lz
            
            # x-periodicity
            dxij[dxij>Lx/2.]  -= Lx
            dxij[dxij<-Lx/2.] += Lx
            
            dij     = np.sqrt(dxij**2. + dzij**2.)
            thetaij = np.arctan2(dzij, dxij)
            
            del xmat, zmat, dxij, dzij
            
            for indr, r in enumerate(rlin[0:-1]):
            
                surf_couche = np.pi * dr * (dr + 2.*r)
                surf_theta  = r * dr * dtheta
                
                condr       = np.logical_and(dij>=r, dij<r+dr)
                thetaij_r   = thetaij[condr]
                
                gr[indr] += np.sum(condr) / (NPPDF * surf_couche)
                for indtheta, theta in enumerate(thetalin[0:-1]):
                    grtheta[indr,indtheta] += np.sum(np.logical_and(thetaij_r>=theta, thetaij_r<theta+dtheta)) / (NPPDF * surf_theta)
                
            del thetaij, condr, thetaij_r
                
        prog = int(100.*(it-SSi)/(ndt-1-SSi))
        if prog%5==0 and prog>prog_old:   prog_old = prog;   print('     ' + str(prog) + ' %')

    gr      /= rho
    grtheta /= rho
    
    file1 = open(Dir+'PDF'+partsTypePDF+'__g_r.txt', "w")
    file1.write('rlin\n')
    for indw in range(len(rlin)):
        file1.write("{:.6f}".format(rlin[indw]) + '      ')
    file1.write('\n\n')
    file1.write('g_r\n')
    for indw in range(len(gr)):
        file1.write("{:.6f}".format(gr[indw]) + '\n')
    file1.close()

    file2 = open(Dir+'PDF'+partsTypePDF+'__g_r_theta.txt', "w")
    file2.write('thetalin\n')
    for indw in range(len(thetalin)):
        file2.write("{:.6f}".format(thetalin[indw]) + '      ')
    file2.write('\n\n')
    file2.write('g_r_theta\n')
    for indw1 in range(grtheta.shape[0]):
        for indw2 in range(grtheta.shape[1]):
            file2.write("{:.6f}".format(grtheta[indw1][indw2]) + '      ')
        file2.write('\n')
    file2.close()





#%% SOME CLUSTERS SNAPSHOTS

def make_SomeClustersSnapshots(Dir, SSi, numSnapshots):
    
    plt.close('all')
    plt.rcParams.update({
      "figure.max_open_warning": 0,
      "text.usetex": True,
      "figure.autolayout": True,
      "font.family": "STIXGeneral",
      "mathtext.fontset": "stix",
      "font.size":        10,
      "xtick.labelsize":  10,
      "ytick.labelsize":  10,
      "patch.linewidth":  .2,
      "lines.markersize":  5,
      "hatch.linewidth":  .2
    })
    plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
    
    hatchStyle = '////////////'

    matplotlib.use('Agg')
    
    baseName     = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile     = Dir + 'data_' + baseName
    rigFile      = Dir + 'rig_'  + baseName
    parFile      = Dir + 'par_'  + baseName
    rigPrimeFile = Dir + 'rigPrime.txt'
    
    sigma             = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))
    
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)
        
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
        
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    
    dummy, a, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    
    dummy, dummy, rx, rz, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)
    
    with open(rigFile) as file:
        fileLines = file.readlines()[2*ndt+5:]
    rigPartsIDs = []
    counter     = -1
    isNewTime   = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                rigPartsIDs.append([])
                isNewTime  = True
                counter   += 1
        else:
            IDs = np.unique([int(item) for item in line.split(",")])
            if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
            rigPartsIDs[counter].append(IDs)
            isNewTime = False
    if len(rigPartsIDs) != ndt:
        sys.exit("ERROR: something's wrong with the reading of rigFile")
    del file, fileLines, line, counter, isNewTime, IDs
    
    with open(rigPrimeFile) as file:
        fileLines = file.readlines()[ndt+3:]
    primeRigPartsIDs = []
    counter          = -1
    isNewTime        = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                primeRigPartsIDs.append([])
                isNewTime  = True
                counter   += 1
        else:
            IDs = np.unique([int(item) for item in line.split(",")])
            if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
            primeRigPartsIDs[counter].append(IDs)
            isNewTime = False
    if len(primeRigPartsIDs) != ndt:
        sys.exit("ERROR: something's wrong with the reading of rigPrimeFile")
    del file, fileLines, line, counter, isNewTime, IDs
    
    a2    = np.max(a)
    newLx = Lx + 2*a2
    newLz = Lz + 2*a2
    
    if not os.path.exists(Dir+"some_snapshots"):
        os.mkdir(Dir+"some_snapshots")
    if os.path.exists(Dir+"some_snapshots/clusters"):
        shutil.rmtree(Dir+"some_snapshots/clusters")
    os.mkdir(Dir+"some_snapshots/clusters")
        
    rangeSomeSnapshots = np.linspace(SSi, ndt-1, numSnapshots, dtype=int)
    
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
    
    print("   >> Generating snapshots")
    
    for it in range(ndt):
        
        if np.isin(it, rangeSomeSnapshots):
            
            print("    - time step " + str(it+1) + " out of " + str(ndt))
            
            RigClustersPartsIDs = rigPartsIDs[it]
            if len(RigClustersPartsIDs) > 0:
                RigClustersPartsIDs = np.concatenate(RigClustersPartsIDs)
                
            primeRigClustersPartsIDs = primeRigPartsIDs[it]
            if len(primeRigClustersPartsIDs) > 0:
                primeRigClustersPartsIDs = np.concatenate(primeRigClustersPartsIDs)
            
            allPartsIDs           = np.array(range(0,NP))
            NoRigClustersPartsIDs = allPartsIDs[np.isin(allPartsIDs,RigClustersPartsIDs)==False]
            RigNoPrimePartsIDs    = allPartsIDs[np.logical_and(np.isin(allPartsIDs,RigClustersPartsIDs)==True,  np.isin(allPartsIDs,primeRigClustersPartsIDs)==False)]
                
            title = r"$\sigma/\sigma_0 =\ $" + str(sigma) + r"$\quad N =\ $" + str(NP) + r"$\quad \phi =\ $" + str(phi) + r"$\quad t =\ $" + '{:.1f}'.format(t[it]) + r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[it])
            ax1.clear()
            ax1.set_title(title)
            for i in NoRigClustersPartsIDs:
                circle = plt.Circle((rx[it][i],rz[it][i]), a[i], facecolor='w', edgecolor='k', zorder=1)
                ax1.add_artist(circle)
            for i in RigNoPrimePartsIDs:
                circle = plt.Circle((rx[it][i],rz[it][i]), a[i], facecolor='w', edgecolor='k', hatch=hatchStyle, zorder=2)
                ax1.add_artist(circle)
            for i in primeRigClustersPartsIDs:
                circle = plt.Circle((rx[it][i],rz[it][i]), a[i], facecolor='#A00000', edgecolor=None, zorder=3)
                ax1.add_artist(circle)
            ax1.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
            ax1.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
            ax1.axis('off')
            ax1.set_aspect('equal')
            fig1.savefig(Dir+"some_snapshots/clusters/"+str(it+1)+".pdf")
            
    plt.close('all')





#%% SOME INTERACTIONS SNAPSHOTS

def make_SomeInteractionsSnapshots(Dir, SSi, numSnapshots):
    
    plt.close('all')
    plt.rcParams.update({
      "figure.max_open_warning": 0,
      "text.usetex": True,
      "figure.autolayout": True,
      "font.family": "STIXGeneral",
      "mathtext.fontset": "stix",
      "font.size":        10,
      "xtick.labelsize":  10,
      "ytick.labelsize":  10,
      "patch.linewidth":  .2,
      "lines.markersize":  5,
      "hatch.linewidth":  .2
    })
    plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
    
    matplotlib.use('Agg')
    
    cmap              = matplotlib.colormaps['gist_rainbow']
    alpha             = 0.75

    hls               = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
    hls[:,1]         *= alpha
    rgb               = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    cmap              = colors.LinearSegmentedColormap.from_list("", rgb)

    maxLineWidth      = 5
    
    baseName          = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile          = Dir + 'data_' + baseName
    intFile           = Dir + 'int_'  + baseName
    parFile           = Dir + 'par_'  + baseName
    
    sigma             = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))
    
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
        
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    
    dummy, a, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    
    dummy, dummy, rx, rz, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)
    
    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData   = []
    counter   = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime  = True
                counter   += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")
    del file, fileLines, line, counter, isNewTime
    
    ip                    = []
    jp                    = []
    nx                    = []
    nz                    = []
    xi                    = []
    normLub               = []
    tanLubX               = []
    tanLubZ               = []
    contState             = []
    normCont              = []
    tanContX              = []
    tanContZ              = []
    normRep               = []
    normInts              = []
    numInts               = []
    maxForces             = []
    
    print("   >> Reading data")
    
    for it in range(ndt):
        
        print("    - time step " + str(it+1) + " out of " + str(ndt))
        
        ip_it, jp_it, nx_it, dummy, nz_it, xi_it, normLub_it, tanLubX_it, dummy, tanLubZ_it, contState_it, normCont_it, tanContX_it, dummy, tanContZ_it, dummy, normRep_it = np.reshape(intData[it], (len(intData[it]), 17)).T
        
        normInts_it = np.abs(normLub_it + normCont_it + normRep_it + np.linalg.norm(np.array([tanLubX_it,tanLubZ_it]),axis=0) + np.linalg.norm(np.array([tanContX_it,tanContZ_it]),axis=0))
        
        ip.append([int(x) for x in ip_it])
        jp.append([int(x) for x in jp_it])
        nx.append(nx_it)
        nz.append(nz_it)
        xi.append(xi_it)
        normLub.append(normLub_it)
        tanLubX.append(tanLubX_it)
        tanLubZ.append(tanLubZ_it)
        contState.append(contState_it)
        normCont.append(normCont_it)
        tanContX.append(tanContX_it)
        tanContZ.append(tanContZ_it)
        normRep.append(normRep_it)
        normInts.append(normInts_it)
        maxForces.append(np.max(normInts_it))
        
    maxForce = np.max(maxForces)
    
    a2    = np.max(a)
    newLx = Lx + 2*a2
    newLz = Lz + 2*a2
    
    if not os.path.exists(Dir+"some_snapshots"):
        os.mkdir(Dir+"some_snapshots")
    if os.path.exists(Dir+"some_snapshots/interactions"):
        shutil.rmtree(Dir+"some_snapshots/interactions")
    os.mkdir(Dir+"some_snapshots/interactions")
    
    rangeSomeSnapshots = np.linspace(SSi, ndt-1, numSnapshots, dtype=int)
    
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
    
    print("   >> Generating snapshots")
    
    for ss in rangeSomeSnapshots:
        
        print("    - time step " + str(ss+1) + " out of " + str(ndt))
        
        allPartsIDs           = np.array(range(0,NP))
        lineWidths            = maxLineWidth * normInts[ss] / maxForce
        colorInts             = np.array(['r'] * numInts[ss], dtype=object)
        contactLess           = np.where(contState[ss]==0)[0]
        frictionLess          = np.where(contState[ss]==1)[0]
        if contactLess.size   > 0: colorInts[contactLess]  = 'tab:cyan'
        if frictionLess.size  > 0: colorInts[frictionLess] = 'g'
        
        title = r"$\sigma/\sigma_0 =\ $" + str(sigma) + r"$\quad N =\ $" + str(NP) + r"$\quad \phi =\ $" + str(phi) + r"$\quad t =\ $" + '{:.1f}'.format(t[it]) + r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[it])
        
        # plot interactions
        ax1.clear()    
        ax1.set_title(title)
        for i in allPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], facecolor='#323232', edgecolor=None)
            ax1.add_artist(circle)
        for i in range(numInts[ss]):
            ipInt = ip[ss][i]
            jpInt = jp[ss][i]
            nij   = np.array([nx[ss][i], nz[ss][i]])
            rij   = nij * (xi[ss][i]+2.) * (a[ipInt]+a[jpInt]) * 0.5
            p1    = np.array([rx[ss][ipInt], rz[ss][ipInt]])
            p2    = p1 + rij
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colorInts[i], linewidth=lineWidths[i])
            if (np.sign(nij[0]) != np.sign(rx[ss][jpInt]-rx[ss][ipInt]) or
                np.sign(nij[1]) != np.sign(rz[ss][jpInt]-rz[ss][ipInt])):   # periodicity
                p3 = np.array([rx[ss][jpInt], rz[ss][jpInt]])
                p4 = p3 - rij
                ax1.plot([p3[0], p4[0]], [p3[1], p4[1]], color=colorInts[i], linewidth=lineWidths[i])
        ax1.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
        ax1.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
        ax1.axis('off')
        ax1.set_aspect('equal')
        fig1.savefig(Dir+"some_snapshots/interactions/"+str(ss+1)+".pdf")
        
    plt.close('all')





#%% CLUSTERS MOVIE

def make_ClustersMovie(Dir):
    
    plt.close('all')
    plt.rcParams.update({
      "figure.max_open_warning": 0,
      "text.usetex": True,
      "figure.autolayout": True,
      "font.family": "STIXGeneral",
      "mathtext.fontset": "stix",
      "font.size":        10,
      "xtick.labelsize":  10,
      "ytick.labelsize":  10,
      "patch.linewidth":  .2,
      "lines.markersize":  5,
      "hatch.linewidth":  .2
    })
    plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
    
    hatchStyle = '////////////'

    matplotlib.use('Agg')
    
    baseName     = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile     = Dir + 'data_' + baseName
    rigFile      = Dir + 'rig_'  + baseName
    parFile      = Dir + 'par_'  + baseName
    rigPrimeFile = Dir + 'rigPrime.txt'
    
    sigma        = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))
    
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)
        
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
        
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    
    dummy, a, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    
    dummy, dummy, rx, rz, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)
    
    with open(rigFile) as file:
        fileLines = file.readlines()[2*ndt+5:]
    rigPartsIDs = []
    counter     = -1
    isNewTime   = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                rigPartsIDs.append([])
                isNewTime  = True
                counter   += 1
        else:
            IDs = np.unique([int(item) for item in line.split(",")])
            if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
            rigPartsIDs[counter].append(IDs)
            isNewTime = False
    if len(rigPartsIDs) != ndt:
        sys.exit("ERROR: something's wrong with the reading of rigFile")
    del file, fileLines, line, counter, isNewTime, IDs
    
    with open(rigPrimeFile) as file:
        fileLines = file.readlines()[ndt+3:]
    primeRigPartsIDs = []
    counter          = -1
    isNewTime        = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                primeRigPartsIDs.append([])
                isNewTime  = True
                counter   += 1
        else:
            IDs = np.unique([int(item) for item in line.split(",")])
            if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
            primeRigPartsIDs[counter].append(IDs)
            isNewTime = False
    if len(primeRigPartsIDs) != ndt:
        sys.exit("ERROR: something's wrong with the reading of rigPrimeFile")
    del file, fileLines, line, counter, isNewTime, IDs
    
    a2    = np.max(a)
    newLx = Lx + 2*a2
    newLz = Lz + 2*a2
    
    if not os.path.exists(Dir+"snapshots"):
        os.mkdir(Dir+"snapshots")
    if os.path.exists(Dir+"snapshots/clusters"):
        shutil.rmtree(Dir+"snapshots/clusters")
    os.mkdir(Dir+"snapshots/clusters")
        
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
    
    print("   >> Generating snapshots")
    
    for it in range(ndt):
        
        print("    - time step " + str(it+1) + " out of " + str(ndt))
        
        RigClustersPartsIDs = rigPartsIDs[it]
        if len(RigClustersPartsIDs) > 0:
            RigClustersPartsIDs = np.concatenate(RigClustersPartsIDs)
            
        primeRigClustersPartsIDs = primeRigPartsIDs[it]
        if len(primeRigClustersPartsIDs) > 0:
            primeRigClustersPartsIDs = np.concatenate(primeRigClustersPartsIDs)
        
        allPartsIDs           = np.array(range(0,NP))
        NoRigClustersPartsIDs = allPartsIDs[np.isin(allPartsIDs,RigClustersPartsIDs)==False]
        RigNoPrimePartsIDs    = allPartsIDs[np.logical_and(np.isin(allPartsIDs,RigClustersPartsIDs)==True,  np.isin(allPartsIDs,primeRigClustersPartsIDs)==False)]
            
        title = r"$\sigma/\sigma_0 =\ $" + str(sigma) + r"$\quad N =\ $" + str(NP) + r"$\quad \phi =\ $" + str(phi) + r"$\quad t =\ $" + '{:.1f}'.format(t[it]) + r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[it])
        ax1.clear()
        ax1.set_title(title)
        for i in NoRigClustersPartsIDs:
            circle = plt.Circle((rx[it][i],rz[it][i]), a[i], facecolor='w', edgecolor='k', zorder=1)
            ax1.add_artist(circle)
        for i in RigNoPrimePartsIDs:
            circle = plt.Circle((rx[it][i],rz[it][i]), a[i], facecolor='w', edgecolor='k', hatch=hatchStyle, zorder=2)
            ax1.add_artist(circle)
        for i in primeRigClustersPartsIDs:
            circle = plt.Circle((rx[it][i],rz[it][i]), a[i], facecolor='#A00000', edgecolor=None, zorder=3)
            ax1.add_artist(circle)
        ax1.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
        ax1.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
        ax1.axis('off')
        ax1.set_aspect('equal')
        fig1.savefig(Dir+"snapshots/clusters/"+str(it+1)+".png", dpi=200)
        
    plt.close('all')





#%% INTERACTIONS MOVIE

def make_InteractionsMovie(Dir):
    
    plt.close('all')
    plt.rcParams.update({
      "figure.max_open_warning": 0,
      "text.usetex": True,
      "figure.autolayout": True,
      "font.family": "STIXGeneral",
      "mathtext.fontset": "stix",
      "font.size":        10,
      "xtick.labelsize":  10,
      "ytick.labelsize":  10,
      "patch.linewidth":  .2,
      "lines.markersize":  5,
      "hatch.linewidth":  .2
    })
    plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
    
    matplotlib.use('Agg')
    
    cmap              = matplotlib.colormaps['gist_rainbow']
    alpha             = 0.75

    hls               = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
    hls[:,1]         *= alpha
    rgb               = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    cmap              = colors.LinearSegmentedColormap.from_list("", rgb)

    maxLineWidth      = 5
    
    baseName          = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile          = Dir + 'data_' + baseName
    intFile           = Dir + 'int_'  + baseName
    parFile           = Dir + 'par_'  + baseName
    
    sigma             = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))
    
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
        
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    
    dummy, a, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    
    dummy, dummy, rx, rz, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)
    
    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData   = []
    counter   = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime  = True
                counter   += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")
    del file, fileLines, line, counter, isNewTime
    
    ip                    = []
    jp                    = []
    nx                    = []
    nz                    = []
    xi                    = []
    normLub               = []
    tanLubX               = []
    tanLubZ               = []
    contState             = []
    normCont              = []
    tanContX              = []
    tanContZ              = []
    normRep               = []
    normInts              = []
    numInts               = []
    maxForces             = []
    
    print("   >> Reading data")
    
    for it in range(ndt):
        
        print("    - time step " + str(it+1) + " out of " + str(ndt))
        
        ip_it, jp_it, nx_it, dummy, nz_it, xi_it, normLub_it, tanLubX_it, dummy, tanLubZ_it, contState_it, normCont_it, tanContX_it, dummy, tanContZ_it, dummy, normRep_it = np.reshape(intData[it], (len(intData[it]), 17)).T
        
        normInts_it = np.abs(normLub_it + normCont_it + normRep_it + np.linalg.norm(np.array([tanLubX_it,tanLubZ_it]),axis=0) + np.linalg.norm(np.array([tanContX_it,tanContZ_it]),axis=0))
        
        ip.append([int(x) for x in ip_it])
        jp.append([int(x) for x in jp_it])
        nx.append(nx_it)
        nz.append(nz_it)
        xi.append(xi_it)
        normLub.append(normLub_it)
        tanLubX.append(tanLubX_it)
        tanLubZ.append(tanLubZ_it)
        contState.append(contState_it)
        normCont.append(normCont_it)
        tanContX.append(tanContX_it)
        tanContZ.append(tanContZ_it)
        normRep.append(normRep_it)
        normInts.append(normInts_it)
        maxForces.append(np.max(normInts_it))
        
    maxForce = np.max(maxForces)
    
    a2    = np.max(a)
    newLx = Lx + 2*a2
    newLz = Lz + 2*a2
    
    if not os.path.exists(Dir+"snapshots"):
        os.mkdir(Dir+"snapshots")
    if os.path.exists(Dir+"snapshots/interactions"):
        shutil.rmtree(Dir+"snapshots/interactions")
    os.mkdir(Dir+"snapshots/interactions")
    
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
    
    print("   >> Generating snapshots")
    
    for ss in range(ndt):
        
        print("    - time step " + str(ss+1) + " out of " + str(ndt))
        
        allPartsIDs           = np.array(range(0,NP))
        lineWidths            = maxLineWidth * normInts[ss] / maxForce
        colorInts             = np.array(['r'] * numInts[ss], dtype=object)
        contactLess           = np.where(contState[ss]==0)[0]
        frictionLess          = np.where(contState[ss]==1)[0]
        if contactLess.size   > 0: colorInts[contactLess]  = 'tab:cyan'
        if frictionLess.size  > 0: colorInts[frictionLess] = 'g'
        
        title = r"$\sigma/\sigma_0 =\ $" + str(sigma) + r"$\quad N =\ $" + str(NP) + r"$\quad \phi =\ $" + str(phi) + r"$\quad t =\ $" + '{:.1f}'.format(t[it]) + r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[it])
        
        # plot interactions
        ax1.clear()    
        ax1.set_title(title)
        for i in allPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], facecolor='#323232', edgecolor=None)
            ax1.add_artist(circle)
        for i in range(numInts[ss]):
            ipInt = ip[ss][i]
            jpInt = jp[ss][i]
            nij   = np.array([nx[ss][i], nz[ss][i]])
            rij   = nij * (xi[ss][i]+2.) * (a[ipInt]+a[jpInt]) * 0.5
            p1    = np.array([rx[ss][ipInt], rz[ss][ipInt]])
            p2    = p1 + rij
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colorInts[i], linewidth=lineWidths[i])
            if (np.sign(nij[0]) != np.sign(rx[ss][jpInt]-rx[ss][ipInt]) or
                np.sign(nij[1]) != np.sign(rz[ss][jpInt]-rz[ss][ipInt])):   # periodicity
                p3 = np.array([rx[ss][jpInt], rz[ss][jpInt]])
                p4 = p3 - rij
                ax1.plot([p3[0], p4[0]], [p3[1], p4[1]], color=colorInts[i], linewidth=lineWidths[i])
        ax1.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
        ax1.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
        ax1.axis('off')
        ax1.set_aspect('equal')
        fig1.savefig(Dir+"snapshots/interactions/"+str(ss+1)+".png", dpi=200)
        
    plt.close('all')





#%% TRIANGULAR WALLS MOVIE

def make_TriangularWallsMovie(Dir, paramsFileName, shearRate):
    
    plt.close('all')
    plt.rcParams.update({
      "figure.max_open_warning": 0,
      "text.usetex": True,
      "figure.autolayout": True,
      "font.family": "STIXGeneral",
      "mathtext.fontset": "stix",
      "font.size":          10,
      "xtick.labelsize":    10,
      "ytick.labelsize":    10,
      "patch.linewidth":    .2,
      "lines.markersize":   5,
      "hatch.linewidth":    .2,
      'savefig.pad_inches': .01,
      'savefig.format':     'png',
      'savefig.bbox':       'tight',
      'savefig.dpi':         200
    })
    plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
    
    matplotlib.use('Agg')
    
    baseName       = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    configFileName = Dir + baseName.removesuffix('_'+paramsFileName+'_rate'+shearRate+'h_shear.dat') + '.dat'
    dataFile       = Dir + 'data_' + baseName
    parFile        = Dir + 'par_'  + baseName
    
    np1            = int(np.genfromtxt(configFileName, comments=None, skip_header=1, max_rows=1)[1])
    np2            = int(np.genfromtxt(configFileName, comments=None, skip_header=1, max_rows=1)[2])

    VF             =     np.genfromtxt(configFileName, comments=None, skip_header=1, max_rows=1)[3]

    Lx             =     np.genfromtxt(configFileName, comments=None, skip_header=1, max_rows=1)[4]
    Lz             =     np.genfromtxt(configFileName, comments=None, skip_header=1, max_rows=1)[6]

    np_wall1       = int(np.genfromtxt(configFileName, comments=None, skip_header=1, max_rows=1)[7])
    np_wall2       = int(np.genfromtxt(configFileName, comments=None, skip_header=1, max_rows=1)[8])

    NP             = np1 + np2

    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=40).transpose()
        
    ndt = len(t)
        
    dummy, dummy, dummy, a = np.loadtxt(configFileName, skiprows=2).transpose()
        
    dummy, dummy, rx, dummy, rz, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile).reshape(ndt,NP+np_wall1+np_wall2,15).transpose(2,0,1)
    
    a2    = np.max(a)
    newLx = Lx + 2*a2
    newLz = Lz + 2*a2
    
    if not os.path.exists(Dir+"snapshots"):
        os.mkdir(Dir+"snapshots")
        
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    
    print("   >> Generating snapshots")
    
    for it in range(ndt):
        
        print("    - time step " + str(it+1) + " out of " + str(ndt))
        
        title = r"$\dot{\gamma} =\ $" + shearRate + r"$\qquad N =\ $" + str(NP) + r"$\qquad \phi =\ $" + str(VF) + r"$\qquad \gamma =\ $" + '{:.2f}'.format(gamma[it])
        ax.clear()
        ax.set_title(title)
        for ip in range(NP):
            circle = plt.Circle((rx[it][ip],rz[it][ip]), a[ip], facecolor='w', edgecolor='k', zorder=1)
            ax.add_artist(circle)
        for ip in range(NP,NP+np_wall1+np_wall2):
            circle = plt.Circle((rx[it][ip],rz[it][ip]), a[ip], facecolor='b', edgecolor='k', zorder=2)
            ax.add_artist(circle)
        ax.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
        ax.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
        ax.axis('off')
        ax.set_aspect('equal')
        fig.savefig(Dir+"snapshots/"+str(it+1))
        
    plt.close('all')
