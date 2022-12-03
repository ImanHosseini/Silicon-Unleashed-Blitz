import matplotlib.pyplot as plt
from subprocess import Popen
import subprocess
import os
import GPUtil
gpuname = GPUtil.getGPUs()[0].name

vs = [1200,10000,16000,20000]
ss = [16,32,40,50,64,80,100,128,140,160,200,256,400,512]

def test(v, s):
    fn = f"./bins/t_{v}_{s}"
    cmnd = ["nvcc",f"-DUR={s}",f"-DVN={v}","h2.cu","-o",fn]
    print(" ".join(cmnd))
    cproc = Popen(cmnd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    cproc.wait()
    cmnd = [fn]
    print(" ".join(cmnd))
    tproc = Popen(cmnd,stdout=subprocess.PIPE)
    tproc.wait()
    ls = tproc.stdout.readlines()
    t = 0
    for l_ in ls:
        l = l_.decode()
        if(l.startswith("[time]")):
            t += float(l.split()[1]) 
    print(t)
    return t

ls = []
for v in vs:
    times = []
    best = 10000000.0
    for s in ss:
        t = test(v,s)
        if t<best:
            best = t
        times.append(t)
    print(f"BEST: {best}")
    if best == 0.0:
        best = 0.0000000001
    times = [x/best for x in times]
    ls.append(times)

for i, l in enumerate(ls):
    plt.plot(ss,l,label=f"V{vs[i]}")

plt.xlabel('W/thread')
plt.ylabel(r'$T/T_{best}$')
plt.title(gpuname)
plt.legend()
plt.savefig("randTest.png")
