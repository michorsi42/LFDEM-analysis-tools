import os
import sys
import glob
import numpy       as     np
from   collections import defaultdict





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





#%% Z & C
### Identifying the average number of contacts and contraints per particle
### (globally or in the contact network)

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





#%% K>=n & C>=n
### Identifying the particles with at least 3,4,6 contacts and particles with at least 6,7,8 constraints

def KC_parts(Dir):

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
    
    F_C6            = []
    F_C7            = []
    F_C8            = []
    F_K3            = []
    F_K4            = []
    F_K6            = []
    C6Clusters      = []
    C6ClustersSizes = []
    C7Clusters      = []
    C7ClustersSizes = []
    C8Clusters      = []
    C8ClustersSizes = []
    K3Clusters      = []
    K3ClustersSizes = []
    K4Clusters      = []
    K4ClustersSizes = []
    K6Clusters      = []
    K6ClustersSizes = []
    
    for it in range(ndt):
        
        ip, jp, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, contState, dummy, dummy, dummy, dummy, dummy, dummy = np.reshape(intData[it], (len(intData[it]), 17)).T
        
        ip            = np.array(ip,        dtype=int)
        jp            = np.array(jp,        dtype=int)
        contState     = np.array(contState, dtype=int)
        frictContacts = np.where(contState> 1)[0]
        stickContacts = np.where(contState==2)[0]
        slideContacts = np.where(contState==3)[0]
        
        partsInCont = [[] for i in range(NP)]
        for i in frictContacts:
            partsInCont[ip[i]].append(jp[i])
            partsInCont[jp[i]].append(ip[i])
        
        numConstraintsPerPart = np.zeros(NP, dtype=int)
        for i in range(stickContacts.size):
            numConstraintsPerPart[ip[stickContacts[i]]] += 2
            numConstraintsPerPart[jp[stickContacts[i]]] += 2
        for i in range(slideContacts.size):
            numConstraintsPerPart[ip[slideContacts[i]]] += 1
            numConstraintsPerPart[jp[slideContacts[i]]] += 1
        
        numFrictContactsPerPart = np.zeros(NP, dtype=int)
        for i in range(frictContacts.size):
            numFrictContactsPerPart[ip[frictContacts[i]]] += 1
            numFrictContactsPerPart[jp[frictContacts[i]]] += 1
        
        C6PartsIDs = np.where(numConstraintsPerPart>=6)[0]
        C7PartsIDs = np.where(numConstraintsPerPart>=7)[0]
        C8PartsIDs = np.where(numConstraintsPerPart>=8)[0]
        
        K3PartsIDs = np.where(numFrictContactsPerPart>=3)[0]
        K4PartsIDs = np.where(numFrictContactsPerPart>=4)[0]
        K6PartsIDs = np.where(numFrictContactsPerPart>=6)[0]
        
        F_C6.append(len(C6PartsIDs))
        F_C7.append(len(C7PartsIDs))
        F_C8.append(len(C8PartsIDs))
        
        F_K3.append(len(K3PartsIDs))
        F_K4.append(len(K4PartsIDs))
        F_K6.append(len(K6PartsIDs))
        
        # C6 clusters
        C6Clusters_it = []
        for i in frictContacts:
            if ip[i] in C6PartsIDs and jp[i] not in C6PartsIDs:
                ip_already_in_C6_cluster = False
                for j in range(len(C6Clusters_it)):
                    if ip[i] in C6Clusters_it[j]:
                        ip_already_in_C6_cluster = True
                        break
                if not ip_already_in_C6_cluster:
                    C6Clusters_it.append([ip[i]])
            elif ip[i] not in C6PartsIDs and jp[i] in C6PartsIDs:
                jp_already_in_C6_cluster = False
                for j in range(len(C6Clusters_it)):
                    if jp[i] in C6Clusters_it[j]:
                        jp_already_in_C6_cluster = True
                        break
                if not jp_already_in_C6_cluster:
                    C6Clusters_it.append([jp[i]])
            elif ip[i] in C6PartsIDs and jp[i] in C6PartsIDs:
                ip_already_in_C6_cluster = False
                jp_already_in_C6_cluster = False
                for j in range(len(C6Clusters_it)):
                    if ip[i] in C6Clusters_it[j] and jp[i] not in C6Clusters_it[j]:
                        ip_already_in_C6_cluster = True
                        jp_already_in_C6_cluster = True
                        C6Clusters_it[j].append(jp[i])
                    elif ip[i] not in C6Clusters_it[j] and jp[i] in C6Clusters_it[j]:
                        ip_already_in_C6_cluster = True
                        jp_already_in_C6_cluster = True
                        C6Clusters_it[j].append(ip[i])
                if not ip_already_in_C6_cluster and not jp_already_in_C6_cluster:
                    C6Clusters_it.append([ip[i], jp[i]])
        C6Clusters_it = list(merge_common(C6Clusters_it))
        if len(C6Clusters_it) > 0:
            if len(np.concatenate(C6Clusters_it)) != len(C6PartsIDs):
                sys.exit("ERROR: something's wrong with C6Clusters_it")
        elif len(C6Clusters_it) == 0 and len(C6PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with C6Clusters_it")
        C6Clusters.append(C6Clusters_it)
        C6ClustersSizes.append([len(C6Clusters_it[i]) for i in range(len(C6Clusters_it))])
        
        # C7 clusters
        C7Clusters_it = []
        for i in frictContacts:
            if ip[i] in C7PartsIDs and jp[i] not in C7PartsIDs:
                ip_already_in_C7_cluster = False
                for j in range(len(C7Clusters_it)):
                    if ip[i] in C7Clusters_it[j]:
                        ip_already_in_C7_cluster = True
                        break
                if not ip_already_in_C7_cluster:
                    C7Clusters_it.append([ip[i]])
            elif ip[i] not in C7PartsIDs and jp[i] in C7PartsIDs:
                jp_already_in_C7_cluster = False
                for j in range(len(C7Clusters_it)):
                    if jp[i] in C7Clusters_it[j]:
                        jp_already_in_C7_cluster = True
                        break
                if not jp_already_in_C7_cluster:
                    C7Clusters_it.append([jp[i]])
            elif ip[i] in C7PartsIDs and jp[i] in C7PartsIDs:
                ip_already_in_C7_cluster = False
                jp_already_in_C7_cluster = False
                for j in range(len(C7Clusters_it)):
                    if ip[i] in C7Clusters_it[j] and jp[i] not in C7Clusters_it[j]:
                        ip_already_in_C7_cluster = True
                        jp_already_in_C7_cluster = True
                        C7Clusters_it[j].append(jp[i])
                    elif ip[i] not in C7Clusters_it[j] and jp[i] in C7Clusters_it[j]:
                        ip_already_in_C7_cluster = True
                        jp_already_in_C7_cluster = True
                        C7Clusters_it[j].append(ip[i])
                if not ip_already_in_C7_cluster and not jp_already_in_C7_cluster:
                    C7Clusters_it.append([ip[i], jp[i]])
        C7Clusters_it = list(merge_common(C7Clusters_it))
        if len(C7Clusters_it) > 0:
            if len(np.concatenate(C7Clusters_it)) != len(C7PartsIDs):
                sys.exit("ERROR: something's wrong with C7Clusters_it")
        elif len(C7Clusters_it) == 0 and len(C7PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with C7Clusters_it")
        C7Clusters.append(C7Clusters_it)
        C7ClustersSizes.append([len(C7Clusters_it[i]) for i in range(len(C7Clusters_it))])
        
        # C8 clusters
        C8Clusters_it = []
        for i in frictContacts:
            if ip[i] in C8PartsIDs and jp[i] not in C8PartsIDs:
                ip_already_in_C8_cluster = False
                for j in range(len(C8Clusters_it)):
                    if ip[i] in C8Clusters_it[j]:
                        ip_already_in_C8_cluster = True
                        break
                if not ip_already_in_C8_cluster:
                    C8Clusters_it.append([ip[i]])
            elif ip[i] not in C8PartsIDs and jp[i] in C8PartsIDs:
                jp_already_in_C8_cluster = False
                for j in range(len(C8Clusters_it)):
                    if jp[i] in C8Clusters_it[j]:
                        jp_already_in_C8_cluster = True
                        break
                if not jp_already_in_C8_cluster:
                    C8Clusters_it.append([jp[i]])
            elif ip[i] in C8PartsIDs and jp[i] in C8PartsIDs:
                ip_already_in_C8_cluster = False
                jp_already_in_C8_cluster = False
                for j in range(len(C8Clusters_it)):
                    if ip[i] in C8Clusters_it[j] and jp[i] not in C8Clusters_it[j]:
                        ip_already_in_C8_cluster = True
                        jp_already_in_C8_cluster = True
                        C8Clusters_it[j].append(jp[i])
                    elif ip[i] not in C8Clusters_it[j] and jp[i] in C8Clusters_it[j]:
                        ip_already_in_C8_cluster = True
                        jp_already_in_C8_cluster = True
                        C8Clusters_it[j].append(ip[i])
                if not ip_already_in_C8_cluster and not jp_already_in_C8_cluster:
                    C8Clusters_it.append([ip[i], jp[i]])
        C8Clusters_it = list(merge_common(C8Clusters_it))
        if len(C8Clusters_it) > 0:
            if len(np.concatenate(C8Clusters_it)) != len(C8PartsIDs):
                sys.exit("ERROR: something's wrong with C8Clusters_it")
        elif len(C8Clusters_it) == 0 and len(C8PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with C8Clusters_it")
        C8Clusters.append(C8Clusters_it)
        C8ClustersSizes.append([len(C8Clusters_it[i]) for i in range(len(C8Clusters_it))])
        
        # K3 clusters
        K3Clusters_it = []
        for i in frictContacts:
            if ip[i] in K3PartsIDs and jp[i] not in K3PartsIDs:
                ip_already_in_K3_cluster = False
                for j in range(len(K3Clusters_it)):
                    if ip[i] in K3Clusters_it[j]:
                        ip_already_in_K3_cluster = True
                        break
                if not ip_already_in_K3_cluster:
                    K3Clusters_it.append([ip[i]])
            elif ip[i] not in K3PartsIDs and jp[i] in K3PartsIDs:
                jp_already_in_K3_cluster = False
                for j in range(len(K3Clusters_it)):
                    if jp[i] in K3Clusters_it[j]:
                        jp_already_in_K3_cluster = True
                        break
                if not jp_already_in_K3_cluster:
                    K3Clusters_it.append([jp[i]])
            elif ip[i] in K3PartsIDs and jp[i] in K3PartsIDs:
                ip_already_in_K3_cluster = False
                jp_already_in_K3_cluster = False
                for j in range(len(K3Clusters_it)):
                    if ip[i] in K3Clusters_it[j] and jp[i] not in K3Clusters_it[j]:
                        ip_already_in_K3_cluster = True
                        jp_already_in_K3_cluster = True
                        K3Clusters_it[j].append(jp[i])
                    elif ip[i] not in K3Clusters_it[j] and jp[i] in K3Clusters_it[j]:
                        ip_already_in_K3_cluster = True
                        jp_already_in_K3_cluster = True
                        K3Clusters_it[j].append(ip[i])
                if not ip_already_in_K3_cluster and not jp_already_in_K3_cluster:
                    K3Clusters_it.append([ip[i], jp[i]])
        K3Clusters_it = list(merge_common(K3Clusters_it))
        if len(K3Clusters_it) > 0:
            if len(np.concatenate(K3Clusters_it)) != len(K3PartsIDs):
                sys.exit("ERROR: something's wrong with K3Clusters_it")
        elif len(K3Clusters_it) == 0 and len(K3PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with K3Clusters_it")
        K3Clusters.append(K3Clusters_it)
        K3ClustersSizes.append([len(K3Clusters_it[i]) for i in range(len(K3Clusters_it))])
        
        # K4 clusters
        K4Clusters_it = []
        for i in frictContacts:
            if ip[i] in K4PartsIDs and jp[i] not in K4PartsIDs:
                ip_already_in_K4_cluster = False
                for j in range(len(K4Clusters_it)):
                    if ip[i] in K4Clusters_it[j]:
                        ip_already_in_K4_cluster = True
                        break
                if not ip_already_in_K4_cluster:
                    K4Clusters_it.append([ip[i]])
            elif ip[i] not in K4PartsIDs and jp[i] in K4PartsIDs:
                jp_already_in_K4_cluster = False
                for j in range(len(K4Clusters_it)):
                    if jp[i] in K4Clusters_it[j]:
                        jp_already_in_K4_cluster = True
                        break
                if not jp_already_in_K4_cluster:
                    K4Clusters_it.append([jp[i]])
            elif ip[i] in K4PartsIDs and jp[i] in K4PartsIDs:
                ip_already_in_K4_cluster = False
                jp_already_in_K4_cluster = False
                for j in range(len(K4Clusters_it)):
                    if ip[i] in K4Clusters_it[j] and jp[i] not in K4Clusters_it[j]:
                        ip_already_in_K4_cluster = True
                        jp_already_in_K4_cluster = True
                        K4Clusters_it[j].append(jp[i])
                    elif ip[i] not in K4Clusters_it[j] and jp[i] in K4Clusters_it[j]:
                        ip_already_in_K4_cluster = True
                        jp_already_in_K4_cluster = True
                        K4Clusters_it[j].append(ip[i])
                if not ip_already_in_K4_cluster and not jp_already_in_K4_cluster:
                    K4Clusters_it.append([ip[i], jp[i]])
        K4Clusters_it = list(merge_common(K4Clusters_it))
        if len(K4Clusters_it) > 0:
            if len(np.concatenate(K4Clusters_it)) != len(K4PartsIDs):
                sys.exit("ERROR: something's wrong with K4Clusters_it")
        elif len(K4Clusters_it) == 0 and len(K4PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with K4Clusters_it")
        K4Clusters.append(K4Clusters_it)
        K4ClustersSizes.append([len(K4Clusters_it[i]) for i in range(len(K4Clusters_it))])
        
        # K6 clusters
        K6Clusters_it = []
        for i in frictContacts:
            if ip[i] in K6PartsIDs and jp[i] not in K6PartsIDs:
                ip_already_in_K6_cluster = False
                for j in range(len(K6Clusters_it)):
                    if ip[i] in K6Clusters_it[j]:
                        ip_already_in_K6_cluster = True
                        break
                if not ip_already_in_K6_cluster:
                    K6Clusters_it.append([ip[i]])
            elif ip[i] not in K6PartsIDs and jp[i] in K6PartsIDs:
                jp_already_in_K6_cluster = False
                for j in range(len(K6Clusters_it)):
                    if jp[i] in K6Clusters_it[j]:
                        jp_already_in_K6_cluster = True
                        break
                if not jp_already_in_K6_cluster:
                    K6Clusters_it.append([jp[i]])
            elif ip[i] in K6PartsIDs and jp[i] in K6PartsIDs:
                ip_already_in_K6_cluster = False
                jp_already_in_K6_cluster = False
                for j in range(len(K6Clusters_it)):
                    if ip[i] in K6Clusters_it[j] and jp[i] not in K6Clusters_it[j]:
                        ip_already_in_K6_cluster = True
                        jp_already_in_K6_cluster = True
                        K6Clusters_it[j].append(jp[i])
                    elif ip[i] not in K6Clusters_it[j] and jp[i] in K6Clusters_it[j]:
                        ip_already_in_K6_cluster = True
                        jp_already_in_K6_cluster = True
                        K6Clusters_it[j].append(ip[i])
                if not ip_already_in_K6_cluster and not jp_already_in_K6_cluster:
                    K6Clusters_it.append([ip[i], jp[i]])
        K6Clusters_it = list(merge_common(K6Clusters_it))
        if len(K6Clusters_it) > 0:
            if len(np.concatenate(K6Clusters_it)) != len(K6PartsIDs):
                sys.exit("ERROR: something's wrong with K6Clusters_it")
        elif len(K6Clusters_it) == 0 and len(K6PartsIDs) != 0:
            sys.exit("ERROR: something's wrong with K6Clusters_it")
        K6Clusters.append(K6Clusters_it)
        K6ClustersSizes.append([len(K6Clusters_it[i]) for i in range(len(K6Clusters_it))])
    
    # C6 files
    FC6File = open(Dir+"F_C6.txt", "w")
    FC6File.write("t                F_C6" + '\n')
    for it in range(ndt):
        FC6File.write('{:.4f}'.format(t[it]) + '      ' + str(F_C6[it])  + '\n')
    FC6File.close()
    C6File = open(Dir+"C6_clusters.txt", "w")
    C6File.write("#C>=3 Clusters Sizes" + '\n')
    for it in range(ndt):
        if len(C6ClustersSizes[it]) == 0:
            C6File.write("0   \n")
        else:
            for i in range(len(C6ClustersSizes[it])):
                C6File.write(str(C6ClustersSizes[it][i]) + '   ')
            C6File.write("\n")
    C6File.write("\n")
    C6File.write("#C>=3 Clusters IDs" + '\n')
    for it in range(ndt):
        C6File.write('#snapshot = ' + str(it) + '\n')
        if len(C6ClustersSizes[it]) == 0:
            C6File.write("0\n")
        else:
            for i in range(len(C6Clusters[it])):
                for j in range(len(C6Clusters[it][i])):
                    if j < len(C6Clusters[it][i])-1:
                        C6File.write(str(C6Clusters[it][i][j]) + ',')
                    else:
                        C6File.write(str(C6Clusters[it][i][j]))
                C6File.write("\n")
    C6File.close()
    
    # C7 files
    FC7File = open(Dir+"F_C7.txt", "w")
    FC7File.write("t                F_C7" + '\n')
    for it in range(ndt):
        FC7File.write('{:.4f}'.format(t[it]) + '      ' + str(F_C7[it])  + '\n')
    FC7File.close()
    C7File = open(Dir+"C7_clusters.txt", "w")
    C7File.write("#C>=3 Clusters Sizes" + '\n')
    for it in range(ndt):
        if len(C7ClustersSizes[it]) == 0:
            C7File.write("0   \n")
        else:
            for i in range(len(C7ClustersSizes[it])):
                C7File.write(str(C7ClustersSizes[it][i]) + '   ')
            C7File.write("\n")
    C7File.write("\n")
    C7File.write("#C>=3 Clusters IDs" + '\n')
    for it in range(ndt):
        C7File.write('#snapshot = ' + str(it) + '\n')
        if len(C7ClustersSizes[it]) == 0:
            C7File.write("0\n")
        else:
            for i in range(len(C7Clusters[it])):
                for j in range(len(C7Clusters[it][i])):
                    if j < len(C7Clusters[it][i])-1:
                        C7File.write(str(C7Clusters[it][i][j]) + ',')
                    else:
                        C7File.write(str(C7Clusters[it][i][j]))
                C7File.write("\n")
    C7File.close()
    
    # C8 files
    FC8File = open(Dir+"F_C8.txt", "w")
    FC8File.write("t                F_C8" + '\n')
    for it in range(ndt):
        FC8File.write('{:.4f}'.format(t[it]) + '      ' + str(F_C8[it])  + '\n')
    FC8File.close()
    C8File = open(Dir+"C8_clusters.txt", "w")
    C8File.write("#C>=3 Clusters Sizes" + '\n')
    for it in range(ndt):
        if len(C8ClustersSizes[it]) == 0:
            C8File.write("0   \n")
        else:
            for i in range(len(C8ClustersSizes[it])):
                C8File.write(str(C8ClustersSizes[it][i]) + '   ')
            C8File.write("\n")
    C8File.write("\n")
    C8File.write("#C>=3 Clusters IDs" + '\n')
    for it in range(ndt):
        C8File.write('#snapshot = ' + str(it) + '\n')
        if len(C8ClustersSizes[it]) == 0:
            C8File.write("0\n")
        else:
            for i in range(len(C8Clusters[it])):
                for j in range(len(C8Clusters[it][i])):
                    if j < len(C8Clusters[it][i])-1:
                        C8File.write(str(C8Clusters[it][i][j]) + ',')
                    else:
                        C8File.write(str(C8Clusters[it][i][j]))
                C8File.write("\n")
    C8File.close()
    
    # K3 files
    FK3File = open(Dir+"F_K3.txt", "w")
    FK3File.write("t                F_K3" + '\n')
    for it in range(ndt):
        FK3File.write('{:.4f}'.format(t[it]) + '      ' + str(F_K3[it])  + '\n')
    FK3File.close()
    K3File = open(Dir+"K3_clusters.txt", "w")
    K3File.write("#C>=3 Clusters Sizes" + '\n')
    for it in range(ndt):
        if len(K3ClustersSizes[it]) == 0:
            K3File.write("0   \n")
        else:
            for i in range(len(K3ClustersSizes[it])):
                K3File.write(str(K3ClustersSizes[it][i]) + '   ')
            K3File.write("\n")
    K3File.write("\n")
    K3File.write("#C>=3 Clusters IDs" + '\n')
    for it in range(ndt):
        K3File.write('#snapshot = ' + str(it) + '\n')
        if len(K3ClustersSizes[it]) == 0:
            K3File.write("0\n")
        else:
            for i in range(len(K3Clusters[it])):
                for j in range(len(K3Clusters[it][i])):
                    if j < len(K3Clusters[it][i])-1:
                        K3File.write(str(K3Clusters[it][i][j]) + ',')
                    else:
                        K3File.write(str(K3Clusters[it][i][j]))
                K3File.write("\n")
    K3File.close()
    
    # K4 files
    FK4File = open(Dir+"F_K4.txt", "w")
    FK4File.write("t                F_K4" + '\n')
    for it in range(ndt):
        FK4File.write('{:.4f}'.format(t[it]) + '      ' + str(F_K4[it])  + '\n')
    FK4File.close()
    K4File = open(Dir+"K4_clusters.txt", "w")
    K4File.write("#C>=3 Clusters Sizes" + '\n')
    for it in range(ndt):
        if len(K4ClustersSizes[it]) == 0:
            K4File.write("0   \n")
        else:
            for i in range(len(K4ClustersSizes[it])):
                K4File.write(str(K4ClustersSizes[it][i]) + '   ')
            K4File.write("\n")
    K4File.write("\n")
    K4File.write("#C>=3 Clusters IDs" + '\n')
    for it in range(ndt):
        K4File.write('#snapshot = ' + str(it) + '\n')
        if len(K4ClustersSizes[it]) == 0:
            K4File.write("0\n")
        else:
            for i in range(len(K4Clusters[it])):
                for j in range(len(K4Clusters[it][i])):
                    if j < len(K4Clusters[it][i])-1:
                        K4File.write(str(K4Clusters[it][i][j]) + ',')
                    else:
                        K4File.write(str(K4Clusters[it][i][j]))
                K4File.write("\n")
    K4File.close()
    
    # K6 files
    FK6File = open(Dir+"F_K6.txt", "w")
    FK6File.write("t                F_K6" + '\n')
    for it in range(ndt):
        FK6File.write('{:.4f}'.format(t[it]) + '      ' + str(F_K6[it])  + '\n')
    FK6File.close()
    K6File = open(Dir+"K6_clusters.txt", "w")
    K6File.write("#C>=3 Clusters Sizes" + '\n')
    for it in range(ndt):
        if len(K6ClustersSizes[it]) == 0:
            K6File.write("0   \n")
        else:
            for i in range(len(K6ClustersSizes[it])):
                K6File.write(str(K6ClustersSizes[it][i]) + '   ')
            K6File.write("\n")
    K6File.write("\n")
    K6File.write("#C>=3 Clusters IDs" + '\n')
    for it in range(ndt):
        K6File.write('#snapshot = ' + str(it) + '\n')
        if len(K6ClustersSizes[it]) == 0:
            K6File.write("0\n")
        else:
            for i in range(len(K6Clusters[it])):
                for j in range(len(K6Clusters[it][i])):
                    if j < len(K6Clusters[it][i])-1:
                        K6File.write(str(K6Clusters[it][i][j]) + ',')
                    else:
                        K6File.write(str(K6Clusters[it][i][j]))
                K6File.write("\n")
    K6File.close()





#%% PAIR DISTRIBUTION FUNCTION
### Computing both g(r) and g(r,theta)
### The user needs to specify both dr and dtheta, as well as the type of particles to consider

### TO DO: FFT

def PDF(Dir, SSi, dr, dtheta, partsTypePDF):
    
    if partsTypePDF != 'all' and partsTypePDF != 'rig' and partsTypePDF != 'rigPrime':
        sys.exit("ERROR: partsType can only be 'all', 'rig', or 'rigPrime'")
    
    baseName     = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile     = Dir + 'data_' + baseName
    parFile      = Dir + 'par_'  + baseName
    rigFile      = Dir + 'rig_'  + baseName
    rigPrimeFile = Dir + 'rigPrime.txt'
    
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])

    t,     gamma, dummy, dummy,  dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, minGap, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy,  dummy, dummy, dummy, dummy, dummy, dummy  \
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
    
    dtheta  *= np.pi / 180
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
            dxij[dzij> Lz/2.] -= np.modf(gamma[it])[0] * Lz
            dzij[dzij> Lz/2.] -= Lz
            dxij[dzij<-Lz/2.] += np.modf(gamma[it])[0] * Lz
            dzij[dzij<-Lz/2.] += Lz
            
            # x-periodicity
            dxij[dxij>Lx/2.]  -= Lx
            dxij[dxij<-Lx/2.] += Lx
            
            dij     = np.sqrt(dxij**2. + dzij**2.)
            thetaij = np.arctan2(dzij, dxij)
            
            del xmat, zmat, dxij, dzij
            
            for indr, r in enumerate(rlin[0:-1]):
            
                surf_couche = np.pi * dr * (dr + 2.*r)
                surf_theta  = surf_couche * dtheta / (2*np.pi)
                
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





#%% CONTACT DISTRIBUTION

def conts_distribution(Dir, SSi):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    parFile  = Dir + 'par_'  + baseName
    intFile  = Dir + 'int_'  + baseName
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
    
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    dummy, dummy, rx, rz, vx, dummy, vz, dummy, omy, dummy, dummy = np.loadtxt(parFile).reshape(ndt,NP,11).transpose(2,0,1)
        
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

    anglesBinsEdges   = np.linspace(0, 360, int(360/5)+1) * np.pi / 180
    anglesBinsCenters = 0.5*(anglesBinsEdges[1:]+anglesBinsEdges[:-1])
    
    anglesDists       = np.zeros(len(anglesBinsCenters))
    normContFDists    = np.zeros(len(anglesBinsCenters))
    tanContFDists     = np.zeros(len(anglesBinsCenters))
    
    anglesDists_t     = np.zeros((ndt-SSi,len(anglesBinsCenters)))
    normContFDists_t  = np.zeros((ndt-SSi,len(anglesBinsCenters)))
    tanContFDists_t   = np.zeros((ndt-SSi,len(anglesBinsCenters)))
    
    for it in range(ndt-SSi):
        
        ip, jp, nx, dummy, nz, xi, normLub, tanLubX, dummy, tanLubZ, contState, normCont, tanContX, dummy, tanContZ, dummy, normRep = np.reshape(intData[it+SSi], (len(intData[it+SSi]), 17)).T
        
        ip            = np.array(ip,        dtype=int)
        jp            = np.array(jp,        dtype=int)
        contState     = np.array(contState, dtype=int)
        frictContacts = np.where(contState>1)[0]
        
        nij           = np.array([nx,nz]).T
        tanContVecs   = np.array([tanContX,tanContZ]).T
        tanCont       = np.linalg.norm(tanContVecs, axis=1)
    
        angles_it     = []
        
        for j in frictContacts:
            alpha1  = np.arctan2(nz[j],nx[j])
            if alpha1 < 0: alpha1 += 2*np.pi
            alpha2  = alpha1 - np.pi
            if alpha2 < 0: alpha2 += 2*np.pi
            binId1  = np.where(np.logical_and(alpha1>=anglesBinsEdges[0:-1], alpha1<=anglesBinsEdges[1:]))[0][0]
            binId2  = np.where(np.logical_and(alpha2>=anglesBinsEdges[0:-1], alpha2<=anglesBinsEdges[1:]))[0][0]
            angles_it.append(alpha1)
            angles_it.append(alpha2)
            normContFDists_t[it][binId1] += normCont[j]
            normContFDists_t[it][binId2] += normCont[j]
            tanContFDists_t[it][binId1]  += np.sign(np.cross(tanContVecs[j],nij[j]))*tanCont[j]
            tanContFDists_t[it][binId2]  += np.sign(np.cross(tanContVecs[j],nij[j]))*tanCont[j]
        if angles_it != []:
            anglesDists_t[it], dummy = np.histogram(angles_it, bins=anglesBinsEdges, density=True)
        del angles_it
        
        Fn0                   = np.mean(normContFDists_t[it])
        normContFDists_t[it] /= Fn0
        tanContFDists_t[it]  /= Fn0
        
    anglesDists    = np.mean(anglesDists_t,    axis=0)
    normContFDists = np.mean(normContFDists_t, axis=0)
    tanContFDists  = np.mean(tanContFDists_t,  axis=0)
    
    np.savetxt(Dir+"contactDistribution_angles.txt",    np.array([anglesBinsCenters, anglesDists]),    delimiter='      ', fmt='%.9f', header='anglesBinsCenters      anglesDists')
    np.savetxt(Dir+"contactDistribution_normForce.txt", np.array([anglesBinsCenters, normContFDists]), delimiter='      ', fmt='%.9f', header='anglesBinsCenters      normContFDists')
    np.savetxt(Dir+"contactDistribution_tanForce.txt",  np.array([anglesBinsCenters, tanContFDists]),  delimiter='      ', fmt='%.9f', header='anglesBinsCenters      tanContFDists')






