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
from   numba             import njit, prange





#%% MERGE COMMON

# function to merge sublists having common elements
# link: https://www.geeksforgeeks.org/python-merge-list-with-common-elements-in-a-list-of-lists/
def merge_common(lists):
    """
    Merge sublists having common elements.
    """
    # Flatten all elements and build a mapping from element to sublist indices
    element_to_indices = {}
    for idx, sublist in enumerate(lists):
        for item in sublist:
            if item not in element_to_indices:
                element_to_indices[item] = set()
            element_to_indices[item].add(idx)
    # Build adjacency list for sublists
    n = len(lists)
    adj = [set() for _ in range(n)]
    for indices in element_to_indices.values():
        indices = list(indices)
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                adj[indices[i]].add(indices[j])
                adj[indices[j]].add(indices[i])
    # Find connected components (clusters of sublists)
    visited = np.zeros(n, dtype=bool)
    result = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            group = set()
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    group.update(lists[node])
                    stack.extend(adj[node] - set([node]))
            result.append(sorted(group))
    return result





#%% RIGID CLUSTERS
### Using the Pebble Game to identify the rigid clusters

def myRigidClusters(Dir):
    """
    Using the Pebble Game to identify the rigid clusters.
    """
    rigidClusterProcessor.rigFileGenerator(Dir, Dir)

    # Read in rig_ files and write n_rigid.csv files
    rigFile = Dir + 'rig_' + os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    rigidClusterSizes, numBonds, clusterIDs = rigidClusterProcessor.rigFileReader(rigFile)

    # Vectorized summation
    n_rigid = np.array([np.sum(np.array(clusterLists)) for clusterLists in rigidClusterSizes], dtype=int)

    np.savetxt(Dir+'F_rig.txt', n_rigid[:, None], delimiter=' ', fmt='%d')





#%% PRIME RIGID CLUSTERS
### Not considering the rigid particles that are in contact with non-rigid particles

def myPrimeRigidClusters(Dir):
    """
    Identify prime rigid clusters (not in contact with non-rigid particles).
    """
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName
    intFile  = Dir + 'int_'  + baseName

    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    t = np.loadtxt(dataFile, skiprows=37).transpose()[0]
    ndt = len(t)

    # Read intFile
    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData = []
    counter = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime = True
                counter += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False

    # Read clustersSizes
    with open(rigFile) as file:
        fileLines = file.readlines()[1:ndt+1]
    clustersSizes = [list(map(int, line.split())) for line in fileLines]

    # Read rigPartsIDs
    with open(rigFile) as file:
        fileLines = file.readlines()[2*ndt+5:]
    rigPartsIDs = []
    counter = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                rigPartsIDs.append([])
                isNewTime = True
                counter += 1
        else:
            IDs = np.unique([int(item) for item in line.split(",")])
            if len(IDs) == 1 and IDs == np.array([0]): IDs = np.array([])
            rigPartsIDs[counter].append(IDs)
            isNewTime = False

    F_prime_rig = np.zeros(ndt, dtype=int)
    primeClusters = []
    primeClustersSizes = []

    for it in range(ndt):
        arr = np.array(intData[it])
        ip = arr[:,0].astype(int)
        jp = arr[:,1].astype(int)
        contState = arr[:,10].astype(int)
        frictContacts = np.where(contState > 1)[0]

        rigPartsIDs_it = rigPartsIDs[it]
        if len(rigPartsIDs_it) > 0:
            rigPartsIDs_it = np.concatenate(rigPartsIDs_it)
        else:
            rigPartsIDs_it = np.array([], dtype=int)

        partsInCont = [[] for _ in range(NP)]
        for i in frictContacts:
            partsInCont[ip[i]].append(jp[i])
            partsInCont[jp[i]].append(ip[i])

        rigPrimePartsIDs = []
        rigPartsSet = set(rigPartsIDs_it)
        for i in rigPartsIDs_it:
            if set(partsInCont[i]).issubset(rigPartsSet):
                rigPrimePartsIDs.append(i)

        F_prime_rig[it] = len(rigPrimePartsIDs)

        # Build prime clusters
        primeClusters_it = []
        for i in frictContacts:
            if ip[i] in rigPrimePartsIDs and jp[i] not in rigPrimePartsIDs:
                if not any(ip[i] in cluster for cluster in primeClusters_it):
                    primeClusters_it.append([ip[i]])
            elif ip[i] not in rigPrimePartsIDs and jp[i] in rigPrimePartsIDs:
                if not any(jp[i] in cluster for cluster in primeClusters_it):
                    primeClusters_it.append([jp[i]])
            elif ip[i] in rigPrimePartsIDs and jp[i] in rigPrimePartsIDs:
                found = False
                for cluster in primeClusters_it:
                    if ip[i] in cluster and jp[i] not in cluster:
                        cluster.append(jp[i])
                        found = True
                    elif jp[i] in cluster and ip[i] not in cluster:
                        cluster.append(ip[i])
                        found = True
                if not found:
                    primeClusters_it.append([ip[i], jp[i]])
        primeClusters_it = list(merge_common(primeClusters_it))

        if len(primeClusters_it) > 0:
            if len(np.concatenate(primeClusters_it)) != len(rigPrimePartsIDs):
                sys.exit("ERROR: something's wrong with clusters_it")
        elif len(primeClusters_it) == 0 and len(rigPrimePartsIDs) != 0:
            sys.exit("ERROR: something's wrong with clusters_it")

        primeClusters.append(primeClusters_it)
        primeClustersSizes.append([len(cluster) for cluster in primeClusters_it])

    # Write outputs
    with open(Dir+"F_prime_rig.txt", "w") as FPrimeFile:
        FPrimeFile.write("t                F'_rig\n")
        for it in range(ndt):
            FPrimeFile.write('{:.4f}      {}\n'.format(t[it], F_prime_rig[it]))

    with open(Dir+"rigPrime.txt", "w") as rigPrimeFile:
        rigPrimeFile.write("#Prime Rigid Clusters Sizes\n")
        for sizes in primeClustersSizes:
            if len(sizes) == 0:
                rigPrimeFile.write("0   \n")
            else:
                rigPrimeFile.write('   '.join(map(str, sizes)) + '\n')
        rigPrimeFile.write("\n#Prime Rigid Clusters IDs\n")
        for it, clusters in enumerate(primeClusters):
            rigPrimeFile.write('#snapshot = {}\n'.format(it))
            if len(clusters) == 0:
                rigPrimeFile.write("0\n")
            else:
                for cluster in clusters:
                    rigPrimeFile.write(','.join(map(str, cluster)) + '\n')





#%% CLUSTER SIZE DISTRIBUTION
### Size = number of rigid particles in a cluster

def cluster_size_distribution(Dir, SSi):
    """
    Compute the cluster size distribution for all and prime rigid clusters.
    """
    
    baseName     = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile     = Dir + 'data_' + baseName
    rigFile      = Dir + 'rig_'  + baseName
    rigPrimeFile = Dir + 'rigPrime.txt'

    t = np.loadtxt(dataFile, skiprows=37, usecols=0)
    ndt = len(t)

    # Read cluster sizes
    clustersSizes = np.loadtxt(rigFile, skiprows=1, max_rows=ndt, dtype=int)
    if clustersSizes.ndim == 1:
        clustersSizes = clustersSizes[:, None]
    clustersSizes = clustersSizes.flatten()
    clustersSizes = clustersSizes[clustersSizes != 0]
    if clustersSizes.size == 0:
        clustersSizes = np.array([0])

    # Read prime cluster sizes
    clustersSizes_prime = np.loadtxt(rigPrimeFile, skiprows=1, max_rows=ndt, dtype=int)
    if clustersSizes_prime.ndim == 1:
        clustersSizes_prime = clustersSizes_prime[:, None]
    clustersSizes_prime = clustersSizes_prime.flatten()
    clustersSizes_prime = clustersSizes_prime[clustersSizes_prime != 0]
    if clustersSizes_prime.size == 0:
        clustersSizes_prime = np.array([0])

    cedges       = np.arange(np.min(clustersSizes)-0.5,       np.max(clustersSizes)+1.5)
    cbins        = np.arange(np.min(clustersSizes),           np.max(clustersSizes)+1)
    cedges_prime = np.arange(np.min(clustersSizes_prime)-0.5, np.max(clustersSizes_prime)+1.5)
    cbins_prime  = np.arange(np.min(clustersSizes_prime),     np.max(clustersSizes_prime)+1)

    Pn,       _ = np.histogram(clustersSizes[SSi:],       cedges,       density=True)
    Pn_prime, _ = np.histogram(clustersSizes_prime[SSi:], cedges_prime, density=True)

    # Write P(n) files
    np.savetxt(Dir+"Pn.txt", np.column_stack((cbins, Pn)), fmt=['%.0f', '%.9f'], header='n      P(n)')
    np.savetxt(Dir+"Pn_even.txt", np.column_stack((cbins[cbins%2==0], Pn[cbins%2==0])), fmt=['%.0f', '%.9f'], header='n      P(n)')
    np.savetxt(Dir+"Pn_odd.txt",  np.column_stack((cbins[cbins%2==1], Pn[cbins%2==1])),  fmt=['%.0f', '%.9f'], header='n      P(n)')

    np.savetxt(Dir+"Pn_prime.txt", np.column_stack((cbins_prime, Pn_prime)), fmt=['%.0f', '%.9f'], header="n'      P(n')")
    np.savetxt(Dir+"Pn_prime_even.txt", np.column_stack((cbins_prime[cbins_prime%2==0], Pn_prime[cbins_prime%2==0])), fmt=['%.0f', '%.9f'], header="n'      P(n')")
    np.savetxt(Dir+"Pn_prime_odd.txt",  np.column_stack((cbins_prime[cbins_prime%2==1], Pn_prime[cbins_prime%2==1])),  fmt=['%.0f', '%.9f'], header="n'      P(n')")





#%% MAX CLUSTER SIZE
### Computing the maximum cluster size

def maxClusterSize(Dir):
    """
    Compute the maximum cluster size for each snapshot.
    """
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName

    # Ensure rigid clusters are computed
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)

    t = np.loadtxt(dataFile, skiprows=37, usecols=0)
    ndt = len(t)

    # Read all cluster sizes at once
    clustersSizes = np.loadtxt(rigFile, skiprows=1, max_rows=ndt, dtype=int)
    if clustersSizes.ndim == 1:
        clustersSizes = clustersSizes[:, None]

    # Vectorized max per row
    maxClustersSize = np.max(clustersSizes, axis=1)

    np.savetxt(Dir+"maxClusterSize.txt", maxClustersSize, fmt='%d')





#%% RIGIDITY PERSISTENCE
### Time autocorrelation of particle rigidity

def rigPers(Dir, SSi, outputVar):
    """
    Time autocorrelation of particle rigidity.
    """
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName

    # Ensure rigid clusters are computed
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)

    t, gamma, *rest = np.loadtxt(dataFile, skiprows=37).transpose()
    ndt = len(t)
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])

    if outputVar == 't':
        delta = t[1] - t[0]
        header = 'Delta t       C'
    elif outputVar == 'gamma':
        delta = gamma[1] - gamma[0]
        header = 'Delta gamma       C'
    else:
        sys.exit("ERROR: there is a problem with outputVar")

    with open(rigFile) as file:
        fileLines = file.readlines()[2*ndt+5:]
    rigPartsIDs = []
    counter = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                rigPartsIDs.append([])
                isNewTime = True
                counter += 1
        else:
            IDs = np.unique([int(item) for item in line.split(",")])
            if len(IDs) == 1 and np.all(IDs == 0):
                IDs = np.array([], dtype=int)
            rigPartsIDs[counter].append(IDs)
            isNewTime = False
    if len(rigPartsIDs) != ndt:
        sys.exit("ERROR: something's wrong with the reading of rigFile")

    isInCluster = np.zeros((ndt-SSi, NP), dtype=bool)
    for it in range(SSi, ndt):
        rigPartsIDs_it = rigPartsIDs[it]
        if len(rigPartsIDs_it) > 0:
            rigPartsIDs_it = np.concatenate(rigPartsIDs_it)
        for ip in rigPartsIDs_it:
            isInCluster[it-SSi, ip] = True

    ntaus = ndt - SSi
    rigPers = np.zeros(ntaus)
    # Vectorized autocorrelation
    isInCluster = isInCluster.T  # shape: (NP, ntaus)
    av = np.mean(isInCluster, axis=1, keepdims=True)
    for tau in range(ntaus):
        prod = (isInCluster[:, :ntaus-tau] - av) * (isInCluster[:, tau:ntaus] - av)
        rigPers[tau] = np.mean(prod)

    with open(Dir+"rigPers.txt", "w") as rigPersFile:
        rigPersFile.write(header + '\n')
        for k in range(ntaus):
            rigPersFile.write(f"{round(delta*k,9)}      {rigPers[k]}\n")





#%% Z & C
### Identifying the average number of contacts and contraints per particle
### (globally or in the contact network)

def Z_C(Dir):
    """
    Identifying the average number of contacts and constraints per particle (globally or in the contact network).
    """
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName

    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    t = np.loadtxt(dataFile, skiprows=37, usecols=0)
    ndt = len(t)

    # Read intFile
    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData = []
    counter = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime = True
                counter += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")

    Z_Znet = np.zeros((ndt, 4))
    C_Cnet = np.zeros((ndt, 4))

    for it in range(ndt):
        arr = np.array(intData[it])
        ip = arr[:, 0].astype(int)
        jp = arr[:, 1].astype(int)
        contState = arr[:, 10].astype(int)
        frictContacts = np.where(contState > 1)[0]
        stickContacts = np.where(contState == 2)[0]
        slideContacts = np.where(contState == 3)[0]

        ConstraintsPerPart = np.zeros(NP, dtype=int)
        numContsPerPart = np.zeros(NP, dtype=int)
        numStickContsPerPart = np.zeros(NP, dtype=int)

        # Vectorized contact counting
        np.add.at(numContsPerPart, ip[frictContacts], 1)
        np.add.at(numContsPerPart, jp[frictContacts], 1)

        np.add.at(ConstraintsPerPart, ip[stickContacts], 2)
        np.add.at(ConstraintsPerPart, jp[stickContacts], 2)
        np.add.at(numStickContsPerPart, ip[stickContacts], 1)
        np.add.at(numStickContsPerPart, jp[stickContacts], 1)

        np.add.at(ConstraintsPerPart, ip[slideContacts], 1)
        np.add.at(ConstraintsPerPart, jp[slideContacts], 1)

        if stickContacts.size > 0:
            nonzero_stick = numStickContsPerPart != 0
            Z_Znet[it, 0] = np.mean(numStickContsPerPart)
            Z_Znet[it, 1] = np.std(numStickContsPerPart)
            Z_Znet[it, 2] = np.mean(numStickContsPerPart[nonzero_stick]) if np.any(nonzero_stick) else 0
            Z_Znet[it, 3] = np.std(numStickContsPerPart[nonzero_stick]) if np.any(nonzero_stick) else 0

        if frictContacts.size > 0:
            nonzero_cont = numContsPerPart != 0
            C_Cnet[it, 0] = np.mean(ConstraintsPerPart)
            C_Cnet[it, 1] = np.std(ConstraintsPerPart)
            C_Cnet[it, 2] = np.mean(ConstraintsPerPart[nonzero_cont]) if np.any(nonzero_cont) else 0
            C_Cnet[it, 3] = np.std(ConstraintsPerPart[nonzero_cont]) if np.any(nonzero_cont) else 0

    np.savetxt(Dir+"Z_Znet.txt", Z_Znet, delimiter='      ', fmt='%.9f', header='mean(Z)      std(Z)      mean(Znet)      std(Znet)')
    np.savetxt(Dir+"C_Cnet.txt", C_Cnet, delimiter='      ', fmt='%.9f', header='mean(C)      std(C)      mean(Cnet)      std(Cnet)')





#%% K>=n & C>=n
### Identifying the particles with at least 2,3,4 contacts and particles with at least 3,4,5 constraints

def KC_parts(Dir):
    """
    Identify particles with at least 2,3,4 contacts and at least 3,4,5 constraints.
    """
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName

    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    t = np.loadtxt(dataFile, skiprows=37, usecols=0)
    ndt = len(t)

    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData = []
    counter = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime = True
                counter += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")

    F_C3, F_C4, F_C5 = [], [], []
    F_K2, F_K3, F_K4 = [], [], []
    C3Clusters, C3ClustersSizes = [], []
    C4Clusters, C4ClustersSizes = [], []
    C5Clusters, C5ClustersSizes = [], []
    K2Clusters, K2ClustersSizes = [], []
    K3Clusters, K3ClustersSizes = [], []
    K4Clusters, K4ClustersSizes = [], []

    for it in range(ndt):
        arr = np.array(intData[it])
        ip = arr[:, 0].astype(int)
        jp = arr[:, 1].astype(int)
        contState = arr[:, 10].astype(int)
        frictContacts = np.where(contState > 1)[0]
        stickContacts = np.where(contState == 2)[0]
        slideContacts = np.where(contState == 3)[0]

        numConstraintsPerPart = np.zeros(NP, dtype=int)
        np.add.at(numConstraintsPerPart, ip[stickContacts], 2)
        np.add.at(numConstraintsPerPart, jp[stickContacts], 2)
        np.add.at(numConstraintsPerPart, ip[slideContacts], 1)
        np.add.at(numConstraintsPerPart, jp[slideContacts], 1)

        numFrictContactsPerPart = np.zeros(NP, dtype=int)
        np.add.at(numFrictContactsPerPart, ip[frictContacts], 1)
        np.add.at(numFrictContactsPerPart, jp[frictContacts], 1)

        C3PartsIDs = np.where(numConstraintsPerPart >= 3)[0]
        C4PartsIDs = np.where(numConstraintsPerPart >= 4)[0]
        C5PartsIDs = np.where(numConstraintsPerPart >= 5)[0]
        K2PartsIDs = np.where(numFrictContactsPerPart >= 2)[0]
        K3PartsIDs = np.where(numFrictContactsPerPart >= 3)[0]
        K4PartsIDs = np.where(numFrictContactsPerPart >= 4)[0]

        F_C3.append(len(C3PartsIDs))
        F_C4.append(len(C4PartsIDs))
        F_C5.append(len(C5PartsIDs))
        F_K2.append(len(K2PartsIDs))
        F_K3.append(len(K3PartsIDs))
        F_K4.append(len(K4PartsIDs))

        # Cluster building (remains as in original, as it's set-based)
        def build_clusters(partsIDs, frictContacts, ip, jp):
            clusters_it = []
            for i in frictContacts:
                if ip[i] in partsIDs and jp[i] not in partsIDs:
                    if not any(ip[i] in cluster for cluster in clusters_it):
                        clusters_it.append([ip[i]])
                elif ip[i] not in partsIDs and jp[i] in partsIDs:
                    if not any(jp[i] in cluster for cluster in clusters_it):
                        clusters_it.append([jp[i]])
                elif ip[i] in partsIDs and jp[i] in partsIDs:
                    found = False
                    for cluster in clusters_it:
                        if ip[i] in cluster and jp[i] not in cluster:
                            cluster.append(jp[i])
                            found = True
                        elif jp[i] in cluster and ip[i] not in cluster:
                            cluster.append(ip[i])
                            found = True
                    if not found:
                        clusters_it.append([ip[i], jp[i]])
            clusters_it = list(merge_common(clusters_it))
            return clusters_it

        # C3 clusters
        C3Clusters_it = build_clusters(C3PartsIDs, frictContacts, ip, jp)
        if len(C3Clusters_it) > 0:
            if len(np.concatenate(C3Clusters_it)) != len(C3PartsIDs):
                sys.exit("ERROR: something's wrong with C3Clusters_it")
        elif len(C3Clusters_it) == 0 and len(C3PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with C3Clusters_it")
        C3Clusters.append(C3Clusters_it)
        C3ClustersSizes.append([len(cluster) for cluster in C3Clusters_it])

        # C4 clusters
        C4Clusters_it = build_clusters(C4PartsIDs, frictContacts, ip, jp)
        if len(C4Clusters_it) > 0:
            if len(np.concatenate(C4Clusters_it)) != len(C4PartsIDs):
                sys.exit("ERROR: something's wrong with C4Clusters_it")
        elif len(C4Clusters_it) == 0 and len(C4PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with C4Clusters_it")
        C4Clusters.append(C4Clusters_it)
        C4ClustersSizes.append([len(cluster) for cluster in C4Clusters_it])

        # C5 clusters
        C5Clusters_it = build_clusters(C5PartsIDs, frictContacts, ip, jp)
        if len(C5Clusters_it) > 0:
            if len(np.concatenate(C5Clusters_it)) != len(C5PartsIDs):
                sys.exit("ERROR: something's wrong with C5Clusters_it")
        elif len(C5Clusters_it) == 0 and len(C5PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with C5Clusters_it")
        C5Clusters.append(C5Clusters_it)
        C5ClustersSizes.append([len(cluster) for cluster in C5Clusters_it])

        # K2 clusters
        K2Clusters_it = build_clusters(K2PartsIDs, frictContacts, ip, jp)
        if len(K2Clusters_it) > 0:
            if len(np.concatenate(K2Clusters_it)) != len(K2PartsIDs):
                sys.exit("ERROR: something's wrong with K2Clusters_it")
        elif len(K2Clusters_it) == 0 and len(K2PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with K2Clusters_it")
        K2Clusters.append(K2Clusters_it)
        K2ClustersSizes.append([len(cluster) for cluster in K2Clusters_it])

        # K3 clusters
        K3Clusters_it = build_clusters(K3PartsIDs, frictContacts, ip, jp)
        if len(K3Clusters_it) > 0:
            if len(np.concatenate(K3Clusters_it)) != len(K3PartsIDs):
                sys.exit("ERROR: something's wrong with K3Clusters_it")
        elif len(K3Clusters_it) == 0 and len(K3PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with K3Clusters_it")
        K3Clusters.append(K3Clusters_it)
        K3ClustersSizes.append([len(cluster) for cluster in K3Clusters_it])

        # K4 clusters
        K4Clusters_it = build_clusters(K4PartsIDs, frictContacts, ip, jp)
        if len(K4Clusters_it) > 0:
            if len(np.concatenate(K4Clusters_it)) != len(K4PartsIDs):
                sys.exit("ERROR: something's wrong with K4Clusters_it")
        elif len(K4Clusters_it) == 0 and len(K4PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with K4Clusters_it")
        K4Clusters.append(K4Clusters_it)
        K4ClustersSizes.append([len(cluster) for cluster in K4Clusters_it])

    # Write output files (same as original)
    def write_clusters_file(filename, t, F, Clusters, ClustersSizes, label):
        with open(filename, "w") as f:
            f.write(f"t                {label}\n")
            for it in range(ndt):
                f.write('{:.4f}      {}\n'.format(t[it], F[it]))
        with open(filename.replace("F_", "").replace(".txt", "_clusters.txt"), "w") as f:
            f.write(f"#{label} Clusters Sizes\n")
            for sizes in ClustersSizes:
                if len(sizes) == 0:
                    f.write("0   \n")
                else:
                    f.write('   '.join(map(str, sizes)) + '\n')
            f.write("\n")
            f.write(f"#{label} Clusters IDs\n")
            for it, clusters in enumerate(Clusters):
                f.write(f'#snapshot = {it}\n')
                if len(clusters) == 0:
                    f.write("0\n")
                else:
                    for cluster in clusters:
                        f.write(','.join(map(str, cluster)) + '\n')

    write_clusters_file(Dir+"F_C3.txt", t, F_C3, C3Clusters, C3ClustersSizes, "C>=3")
    write_clusters_file(Dir+"F_C4.txt", t, F_C4, C4Clusters, C4ClustersSizes, "C>=4")
    write_clusters_file(Dir+"F_C5.txt", t, F_C5, C5Clusters, C5ClustersSizes, "C>=5")
    write_clusters_file(Dir+"F_K2.txt", t, F_K2, K2Clusters, K2ClustersSizes, "K>=2")
    write_clusters_file(Dir+"F_K3.txt", t, F_K3, K3Clusters, K3ClustersSizes, "K>=3")
    write_clusters_file(Dir+"F_K4.txt", t, F_K4, K4Clusters, K4ClustersSizes, "K>=4")





#%% PAIR DISTRIBUTION FUNCTION
### Computing both g(r) and g(r,theta)
### The user needs to specify both dr and dtheta, as well as the type of particles to consider

@njit(parallel=True)
def accumulate_gr_grtheta(dxij, dzij, rlin, thetalin, gr, grtheta, NPPDF, gamma, Lx, Lz, dr, dtheta):
    dij     = np.sqrt(dxij**2 + dzij**2)
    thetaij = np.arctan2(dzij, dxij)
    for indr in prange(len(rlin)-1):
        r = rlin[indr]
        surf_couche = np.pi * dr * (dr + 2.*r)
        surf_theta  = surf_couche * dtheta / (2*np.pi)
        condr = (dij >= r) & (dij < r+dr)
        thetaij_r = thetaij[condr]
        gr[indr] += np.sum(condr) / (NPPDF * surf_couche)
        for indtheta in range(len(thetalin)-1):
            theta = thetalin[indtheta]
            grtheta[indr, indtheta] += np.sum((thetaij_r >= theta) & (thetaij_r < theta+dtheta)) / (NPPDF * surf_theta)

def PDF(Dir, SSi, dr, dtheta, partsTypePDF):
    """
    Compute both g(r) and g(r,theta) for specified particle types.
    """
    
    if partsTypePDF not in ('all', 'rig', 'rigPrime'):
        sys.exit("ERROR: partsType can only be 'all', 'rig', or 'rigPrime'")

    baseName     = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile     = Dir + 'data_' + baseName
    parFile      = Dir + 'par_'  + baseName
    rigFile      = Dir + 'rig_'  + baseName
    rigPrimeFile = Dir + 'rigPrime.txt'

    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    t, gamma, *rest = np.loadtxt(dataFile, skiprows=37).transpose()
    ndt = len(t)
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    dummy, a, *rest = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    dummy, dummy, rx, rz, vx, dummy, vz, dummy, omy, dummy, dummy = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)
    minGap = np.loadtxt(dataFile, skiprows=37, usecols=13)

    if partsTypePDF == 'all':
        partsIDs = np.arange(NP)
    else:
        if partsTypePDF == 'rig':
            filename, offset = rigFile, 2*ndt+5
        else:
            filename, offset = rigPrimeFile, ndt+3
        with open(filename) as file:
            fileLines = file.readlines()[offset:]
        partsIDs = []
        counter = -1
        isNewTime = False
        for line in fileLines:
            if "#" in line:
                if not isNewTime:
                    partsIDs.append([])
                    isNewTime = True
                    counter += 1
            else:
                IDs = np.unique([int(item) for item in line.split(",")])
                if len(IDs) == 1 and np.all(IDs == 0): IDs = np.array([], dtype=int)
                partsIDs[counter].append(IDs)
                isNewTime = False
        if len(partsIDs) != ndt:
            sys.exit(f"ERROR: something's wrong with the reading of {filename}")

    dtheta  *= np.pi / 180
    rmin     = np.min(a) + np.min(minGap)
    rmax     = np.max([Lx, Lz]) / 2.
    thetamin = -np.pi
    thetamax =  np.pi
    rlin     = np.arange(rmin,     rmax+dr,         dr)
    thetalin = np.arange(thetamin, thetamax+dtheta, dtheta)

    gr      = np.zeros(len(rlin)-1)
    grtheta = np.zeros((len(rlin)-1, len(thetalin)-1))
    surf    = Lx * Lz
    rho     = 0

    for it in range(SSi, ndt):
        if partsTypePDF == 'all':
            partsIDs_it = partsIDs
        else:
            partsIDs_it = partsIDs[it]
            if len(partsIDs_it) > 0:
                partsIDs_it = np.array([int(i) for i in np.concatenate(partsIDs_it)])
            else:
                partsIDs_it = np.array([], dtype=int)

        NPPDF = len(partsIDs_it)
        rho  += NPPDF / surf

        if NPPDF > 0:
            xp = rx[it][partsIDs_it]
            zp = rz[it][partsIDs_it]
            dxij = xp[:, None] - xp
            dzij = zp[:, None] - zp

            # z-periodicity (Lees Edwards)
            dxij[dzij > Lz/2.] -= np.modf(gamma[it])[0] * Lz
            dzij[dzij > Lz/2.] -= Lz
            dxij[dzij < -Lz/2.] += np.modf(gamma[it])[0] * Lz
            dzij[dzij < -Lz/2.] += Lz

            # x-periodicity
            dxij[dxij > Lx/2.]  -= Lx
            dxij[dxij < -Lx/2.] += Lx

            accumulate_gr_grtheta(dxij, dzij, rlin, thetalin, gr, grtheta, NPPDF, gamma[it], Lx, Lz, dr, dtheta)

    rho = max(rho, 1e-12)  # avoid division by zero
    gr      /= rho
    grtheta /= rho

    np.savetxt(Dir+f'PDF{partsTypePDF}__g_r.txt', np.column_stack((rlin[:-1], gr)), fmt='%.6f', header='r      g(r)')
    with open(Dir+f'PDF{partsTypePDF}__g_r_theta.txt', "w") as file2:
        file2.write('thetalin\n')
        for val in thetalin:
            file2.write(f"{val:.6f}      ")
        file2.write('\n\ng_r_theta\n')
        for row in grtheta:
            file2.write(' '.join(f"{val:.6f}" for val in row) + '\n')





#%% CONTACT DISTRIBUTION

@njit
def accumulate_contact_bins(nx, nz, tanContX, tanContZ, normCont, frictContacts, anglesBinsEdges, normContFDists, tanContFDists):
    for idx in frictContacts:
        alpha1 = np.arctan2(nz[idx], nx[idx])
        if alpha1 < 0: alpha1 += 2*np.pi
        alpha2 = alpha1 - np.pi
        if alpha2 < 0: alpha2 += 2*np.pi
        binId1 = np.searchsorted(anglesBinsEdges, alpha1, side='right') - 1
        binId2 = np.searchsorted(anglesBinsEdges, alpha2, side='right') - 1
        normContFDists[binId1] += normCont[idx]
        normContFDists[binId2] += normCont[idx]
        tanVec = np.array([tanContX[idx], tanContZ[idx]])
        nij = np.array([nx[idx], nz[idx]])
        tanSign1 = np.sign(np.cross(tanVec, nij))
        tanNorm = np.sqrt(tanContX[idx]**2 + tanContZ[idx]**2)
        tanContFDists[binId1] += tanSign1 * tanNorm
        tanContFDists[binId2] += tanSign1 * tanNorm

def conts_distribution(Dir, SSi):
    """
    Compute the angular distribution of contacts and forces.
    """
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    parFile  = Dir + 'par_'  + baseName
    intFile  = Dir + 'int_'  + baseName

    t = np.loadtxt(dataFile, skiprows=37, usecols=0)
    ndt = len(t)
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    dummy, dummy, rx, rz, vx, dummy, vz, dummy, omy, dummy, dummy = np.loadtxt(parFile).reshape(ndt, NP, 11).transpose(2, 0, 1)

    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData = []
    counter = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime = True
                counter += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")

    anglesBinsEdges   = np.linspace(0, 360, int(360/5)+1) * np.pi / 180
    anglesBinsCenters = 0.5 * (anglesBinsEdges[1:] + anglesBinsEdges[:-1])

    anglesDists_t       = np.zeros((ndt-SSi, len(anglesBinsCenters)))
    normContFDists_t    = np.zeros((ndt-SSi, len(anglesBinsCenters)))
    tanContFDists_t     = np.zeros((ndt-SSi, len(anglesBinsCenters)))

    for it in range(ndt-SSi):
        arr = np.array(intData[it+SSi])
        ip = arr[:, 0].astype(int)
        jp = arr[:, 1].astype(int)
        nx = arr[:, 2]
        nz = arr[:, 4]
        tanContX = arr[:, 11]
        tanContZ = arr[:, 13]
        contState = arr[:, 10].astype(int)
        normCont = arr[:, 12]
        frictContacts = np.where(contState > 1)[0]

        # Numba-accelerated binning
        accumulate_contact_bins(nx, nz, tanContX, tanContZ, normCont, frictContacts, anglesBinsEdges, normContFDists_t[it], tanContFDists_t[it])

        # Histogram for angles
        angles_it = []
        for j in frictContacts:
            alpha1 = np.arctan2(nz[j], nx[j])
            if alpha1 < 0: alpha1 += 2*np.pi
            alpha2 = alpha1 - np.pi
            if alpha2 < 0: alpha2 += 2*np.pi
            angles_it.extend([alpha1, alpha2])
        if angles_it:
            anglesDists_t[it], _ = np.histogram(angles_it, bins=anglesBinsEdges, density=True)

        Fn0 = np.mean(normContFDists_t[it]) if np.mean(normContFDists_t[it]) != 0 else 1.0
        normContFDists_t[it] /= Fn0
        tanContFDists_t[it]  /= Fn0

    anglesDists    = np.mean(anglesDists_t,    axis=0)
    normContFDists = np.mean(normContFDists_t, axis=0)
    tanContFDists  = np.mean(tanContFDists_t,  axis=0)

    np.savetxt(Dir+"contactDistribution_angles.txt",    np.column_stack((anglesBinsCenters, anglesDists)),    delimiter='      ', fmt='%.9f', header='anglesBinsCenters      anglesDists')
    np.savetxt(Dir+"contactDistribution_normForce.txt", np.column_stack((anglesBinsCenters, normContFDists)), delimiter='      ', fmt='%.9f', header='anglesBinsCenters      normContFDists')
    np.savetxt(Dir+"contactDistribution_tanForce.txt",  np.column_stack((anglesBinsCenters, tanContFDists)),  delimiter='      ', fmt='%.9f', header='anglesBinsCenters      tanContFDists')





#%% SOME CLUSTERS SNAPSHOTS
### Plotting some rigid clusters snapshots

def make_SomeClustersSnapshots(Dir, SSi, numSnapshots):
    """
    Plot some rigid clusters snapshots.
    """
    
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

    sigma = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))

    if not os.path.exists(rigFile):
        myRigidClusters(Dir)

    t, gamma, *rest = np.loadtxt(dataFile, skiprows=37).transpose()
    ndt = len(t)
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    dummy, a, *rest = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    dummy, dummy, rx, rz, *rest = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)

    # Read rigid and prime rigid parts IDs
    def read_parts_ids(filename, offset):
        with open(filename) as file:
            fileLines = file.readlines()[offset:]
        partsIDs = []
        counter = -1
        isNewTime = False
        for line in fileLines:
            if "#" in line:
                if not isNewTime:
                    partsIDs.append([])
                    isNewTime = True
                    counter += 1
            else:
                IDs = np.unique([int(item) for item in line.split(",")])
                if len(IDs) == 1 and np.all(IDs == 0): IDs = np.array([])
                partsIDs[counter].append(IDs)
                isNewTime = False
        if len(partsIDs) != ndt:
            sys.exit("ERROR: something's wrong with the reading of {}".format(filename))
        return partsIDs

    rigPartsIDs = read_parts_ids(rigFile, 2*ndt+5)
    primeRigPartsIDs = read_parts_ids(rigPrimeFile, ndt+3)

    a2    = np.max(a)
    newLx = Lx + 2*a2
    newLz = Lz + 2*a2

    outdir = Dir+"some_snapshots/clusters"
    if not os.path.exists(Dir+"some_snapshots"):
        os.mkdir(Dir+"some_snapshots")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    rangeSomeSnapshots = np.linspace(SSi, ndt-1, numSnapshots, dtype=int)
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))

    print("   >> Generating snapshots")

    for it in range(ndt):
        if np.isin(it, rangeSomeSnapshots):
            print(f"    - time step {it+1} out of {ndt}")

            RigClustersPartsIDs = np.concatenate(rigPartsIDs[it]) if len(rigPartsIDs[it]) > 0 else np.array([], dtype=int)
            primeRigClustersPartsIDs = np.concatenate(primeRigPartsIDs[it]) if len(primeRigPartsIDs[it]) > 0 else np.array([], dtype=int)

            allPartsIDs = np.arange(NP)
            NoRigClustersPartsIDs = allPartsIDs[~np.isin(allPartsIDs, RigClustersPartsIDs)]
            RigNoPrimePartsIDs = allPartsIDs[np.logical_and(np.isin(allPartsIDs, RigClustersPartsIDs), ~np.isin(allPartsIDs, primeRigClustersPartsIDs))]

            title = (r"$\sigma/\sigma_0 =\ $" + str(sigma) + r"$\quad N =\ $" + str(NP) +
                     r"$\quad \phi =\ $" + str(phi) + r"$\quad t =\ $" + '{:.1f}'.format(t[it]) +
                     r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[it]))
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
            fig1.savefig(f"{outdir}/{it+1}.pdf")

    plt.close('all')





#%% SOME INTERACTIONS SNAPSHOTS
### Plotting some interactions snapshots

def make_SomeInteractionsSnapshots(Dir, SSi, numSnapshots):
    """
    Plot some interactions snapshots.
    """
    
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

    cmap  = matplotlib.colormaps['gist_rainbow']
    alpha = 0.75
    hls   = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
    hls[:,1] *= alpha
    rgb   = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    cmap  = colors.LinearSegmentedColormap.from_list("", rgb)
    maxLineWidth = 5

    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName
    parFile  = Dir + 'par_'  + baseName

    sigma = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))

    t, gamma, *rest = np.loadtxt(dataFile, skiprows=37).transpose()
    ndt = len(t)
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    dummy, a, *rest = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    dummy, dummy, rx, rz, *rest = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)

    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData = []
    counter = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime = True
                counter += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")

    ip, jp, nx, nz, xi, normLub, tanLubX, tanLubZ, contState, normCont, tanContX, tanContZ, normRep, normInts, numInts, maxForces = ([] for _ in range(16))

    print("   >> Reading data")

    for it in range(ndt):
        ip_it, jp_it, nx_it, dummy, nz_it, xi_it, normLub_it, tanLubX_it, dummy, tanLubZ_it, contState_it, normCont_it, tanContX_it, dummy, tanContZ_it, dummy, normRep_it = np.reshape(intData[it], (len(intData[it]), 17)).T
        normInts_it = np.abs(normLub_it + normCont_it + normRep_it +
                             np.linalg.norm(np.array([tanLubX_it, tanLubZ_it]), axis=0) +
                             np.linalg.norm(np.array([tanContX_it, tanContZ_it]), axis=0))
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
        numInts.append(len(ip_it))
        maxForces.append(np.max(normInts_it))

    maxForce = np.max(maxForces)
    a2    = np.max(a)
    newLx = Lx + 2*a2
    newLz = Lz + 2*a2

    outdir = Dir+"some_snapshots/interactions"
    if not os.path.exists(Dir+"some_snapshots"):
        os.mkdir(Dir+"some_snapshots")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    rangeSomeSnapshots = np.linspace(SSi, ndt-1, numSnapshots, dtype=int)
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))

    print("   >> Generating snapshots")

    for ss in rangeSomeSnapshots:
        print(f"    - time step {ss+1} out of {ndt}")

        allPartsIDs = np.arange(NP)
        lineWidths = maxLineWidth * normInts[ss] / maxForce
        colorInts = np.array(['r'] * numInts[ss], dtype=object)
        contactLess = np.where(contState[ss]==0)[0]
        frictionLess = np.where(contState[ss]==1)[0]
        if contactLess.size > 0: colorInts[contactLess] = 'tab:cyan'
        if frictionLess.size > 0: colorInts[frictionLess] = 'g'

        title = (r"$\sigma/\sigma_0 =\ $" + str(sigma) + r"$\quad N =\ $" + str(NP) +
                 r"$\quad \phi =\ $" + str(phi) + r"$\quad t =\ $" + '{:.1f}'.format(t[ss]) +
                 r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[ss]))
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
        fig1.savefig(f"{outdir}/{ss+1}.png", dpi=200)

    plt.close('all')





#%% CLUSTERS MOVIE
### Plotting the rigid clusters snapshots to generate a movie

def make_ClustersMovie(Dir):
    """
    Plot the rigid clusters snapshots to generate a movie.
    """
    
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

    sigma = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))

    if not os.path.exists(rigFile):
        myRigidClusters(Dir)

    t, gamma, *rest = np.loadtxt(dataFile, skiprows=37).transpose()
    ndt = len(t)
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    dummy, a, *rest = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    dummy, dummy, rx, rz, *rest = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)

    # Read rigid and prime rigid parts IDs
    def read_parts_ids(filename, offset):
        with open(filename) as file:
            fileLines = file.readlines()[offset:]
        partsIDs = []
        counter = -1
        isNewTime = False
        for line in fileLines:
            if "#" in line:
                if not isNewTime:
                    partsIDs.append([])
                    isNewTime = True
                    counter += 1
            else:
                IDs = np.unique([int(item) for item in line.split(",")])
                if len(IDs) == 1 and np.all(IDs == 0): IDs = np.array([])
                partsIDs[counter].append(IDs)
                isNewTime = False
        if len(partsIDs) != ndt:
            sys.exit("ERROR: something's wrong with the reading of {}".format(filename))
        return partsIDs

    rigPartsIDs = read_parts_ids(rigFile, 2*ndt+5)
    primeRigPartsIDs = read_parts_ids(rigPrimeFile, ndt+3)

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
        print(f"    - time step {it+1} out of {ndt}")

        RigClustersPartsIDs = np.concatenate(rigPartsIDs[it]) if len(rigPartsIDs[it]) > 0 else np.array([], dtype=int)
        primeRigClustersPartsIDs = np.concatenate(primeRigPartsIDs[it]) if len(primeRigPartsIDs[it]) > 0 else np.array([], dtype=int)

        allPartsIDs = np.arange(NP)
        NoRigClustersPartsIDs = allPartsIDs[~np.isin(allPartsIDs, RigClustersPartsIDs)]
        RigNoPrimePartsIDs = allPartsIDs[np.logical_and(np.isin(allPartsIDs, RigClustersPartsIDs), ~np.isin(allPartsIDs, primeRigClustersPartsIDs))]

        title = (r"$\sigma/\sigma_0 =\ $" + str(sigma) + r"$\quad N =\ $" + str(NP) +
                 r"$\quad \phi =\ $" + str(phi) + r"$\quad t =\ $" + '{:.1f}'.format(t[it]) +
                 r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[it]))
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
### Plotting the interactions snapshots to generate a movie

def make_InteractionsMovie(Dir):
    """
    Plot the interactions snapshots to generate a movie.
    """
    
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

    cmap  = matplotlib.colormaps['gist_rainbow']
    alpha = 0.75
    hls   = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
    hls[:,1] *= alpha
    rgb   = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    cmap  = colors.LinearSegmentedColormap.from_list("", rgb)
    maxLineWidth = 5

    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName
    parFile  = Dir + 'par_'  + baseName

    sigma = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))

    t, gamma, *rest = np.loadtxt(dataFile, skiprows=37).transpose()
    ndt = len(t)
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    dummy, a, *rest = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    dummy, dummy, rx, rz, *rest = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)

    with open(intFile) as file:
        fileLines = file.readlines()[20:]
    intData = []
    counter = -1
    isNewTime = False
    for line in fileLines:
        if "#" in line:
            if not isNewTime:
                intData.append([])
                isNewTime = True
                counter += 1
        else:
            intData[counter].append([float(item) for item in line.split()])
            isNewTime = False
    if len(intData) != ndt:
        sys.exit("ERROR: something's wrong with the reading of intFile")

    ip, jp, nx, nz, xi, normLub, tanLubX, tanLubZ, contState, normCont, tanContX, tanContZ, normRep, normInts, numInts, maxForces = ([] for _ in range(16))

    print("   >> Reading data")

    for it in range(ndt):
        ip_it, jp_it, nx_it, dummy, nz_it, xi_it, normLub_it, tanLubX_it, dummy, tanLubZ_it, contState_it, normCont_it, tanContX_it, dummy, tanContZ_it, dummy, normRep_it = np.reshape(intData[it], (len(intData[it]), 17)).T
        normInts_it = np.abs(normLub_it + normCont_it + normRep_it +
                             np.linalg.norm(np.array([tanLubX_it, tanLubZ_it]), axis=0) +
                             np.linalg.norm(np.array([tanContX_it, tanContZ_it]), axis=0))
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
        numInts.append(len(ip_it))
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
        print(f"    - time step {ss+1} out of {ndt}")

        allPartsIDs = np.arange(NP)
        lineWidths = maxLineWidth * normInts[ss] / maxForce
        colorInts = np.array(['r'] * numInts[ss], dtype=object)
        contactLess = np.where(contState[ss]==0)[0]
        frictionLess = np.where(contState[ss]==1)[0]
        if contactLess.size > 0: colorInts[contactLess] = 'tab:cyan'
        if frictionLess.size > 0: colorInts[frictionLess] = 'g'

        title = (r"$\sigma/\sigma_0 =\ $" + str(sigma) + r"$\quad N =\ $" + str(NP) +
                 r"$\quad \phi =\ $" + str(phi) + r"$\quad t =\ $" + '{:.1f}'.format(t[ss]) +
                 r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[ss]))
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





