import subprocess
import os

N = 126
IMGNAME = "blitz"
def dockrun(idx):
    name = f'c{idx}'
    cmnds = ['docker','run','--network','host']
    cmnds += ['-e',f'DNAME={name}']
    cmnds += ['-e',f'DINF={idx},{N}']
    cmnds += ['--name',name,'-v',f'/home/iman/ds2.0/dataX_{idx}:/datac','--security-opt','seccomp=unconfined','-i',IMGNAME]
    return cmnds

# p = os.spawnlp(os.P_NOWAIT, 'python3', 'monitor.py')
# os.spawnl(os.P_NOWAIT, 'python3','monitor.py')
# print(p)


for i in range(N):
    cmnds = dockrun(i)
    print(" ".join(cmnds))
    subprocess.call(" ".join(cmnds)+" &",shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)