# import apt
# import apt.package
# from apt.package import Version
import os
import subprocess
from glob import glob
from parser import handlefile

import socket
from struct import pack
from threading import Thread
print("HELLO")
NAME = "C"
DINF = "0,0"

if "DNAME" in os.environ:
    NAME = os.environ['DNAME']
if "DINF" in os.environ:
    DINF = os.environ['DINF'] 

idx, tcnt = DINF.split(",")
idx = int(idx)
tcnt = int(tcnt)

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432+1        # The port used by the server

def recvstr(sock,n):
    data = bytearray()
    while len(data) < n:
        chnk = sock.recv(n-len(data))
        if not chnk:
            return None
        data.extend(chnk)
    return data.decode('ascii')

class NetC(Thread):
    def __init__(self):
        Thread.__init__(self)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST,PORT))
        self.sock = sock
        self.sendmsg(f"H:{NAME}")

    def recvmsg(self,zero=False):
        global lstt
        data = self.sock.recv(4) 
        l = int.from_bytes(data,'little')
        if l == 0:
            return None
        msg = recvstr(self.sock,l)
        return msg

    def sendmsg(self,msg):
        encd = msg.encode('ascii')
        n = len(encd)
        hd = pack('=i',n)
        try:
            self.sock.sendall(hd)
            self.sock.sendall(encd)
        except:
            print("[BLITZ] connection error")
            return

# for k in ks:
#     pkg : apt.Package = cache[k]
#     ver : apt.package.Version = pkg.versions[0]
#     deps = ver.get_dependencies('Depends')  
#     if len(deps)>1:
#         print(deps)

EXCL_DIRS = set([".git","contrib","debian"])
def mk_src_tbl(hd):
    stbl = dict()
    dups = set()
    for root, dirs, files in os.walk(hd):
        for baddir in EXCL_DIRS.intersection(set(dirs)):
            dirs.remove(baddir)
        src_files = list(filter(lambda x: x.endswith((".f90",".f95",".f03",".f")),files))
        # make sure no CPP taint!
        # cxx_src_files = list(filter(lambda x: x.endswith(".cc") or x.endswith(".hpp") or x.endswith(".cpp"),files))
        # if len(cxx_src_files) > 0:
        #     return dict()
        if len(src_files) == 0:
            continue
        for src in src_files:
            src_dir = os.path.sep.join([root,src])
            hd, tl = os.path.split(src_dir)
            # forego src files with same name
            if tl in stbl:
                dups.add(tl)
                del stbl[tl]
            elif tl not in dups:
                stbl[tl] = src_dir
    return stbl
# stbl = mk_src_tbl("/home/zlibc")
DDIR = "/datac/"
for p in [f"{DDIR}{x}/" for x in ["c","f"]]:
    if not os.path.exists(p):
        os.mkdir(p)
print("A")
srcf_c = open(f"{DDIR}c/train.src","a")
tgtf_c = open(f"{DDIR}c/train.tgt","a")
srcf_f = open(f"{DDIR}f/train.src","a")
tgtf_f = open(f"{DDIR}f/train.tgt","a")
print("B")
tbls_f = open(f"{DDIR}tbls.txt","a")
log = open(f"{DDIR}log.txt","a")
print("C")
env = os.environ
env['LD_PRELOAD'] = "/home/blitz/unlink/myunlink.so"
# ks = ['zlibc']
kdir = ""
stbl = dict()
# ks = ['zlibc']
netc = NetC()
total_smpl = 0

netc.sendmsg("V:V")
k = netc.recvmsg().strip() 
cntC, cntF = 0, 0

while k != "ZXYXZ":
    log.write(f"PKG: {k}\n")
    os.mkdir(f"/home/{k}")
    try:
        # ret = subprocess.run(['apt-get','source','--compile',k],cwd=f"/home/{k}",env=env)
        netc.sendmsg(f"I:[purple]<{k}>[/purple] fetch started")
        ret = subprocess.run(['apt-get','source',k],cwd=f"/home/{k}",env=env)
        netc.sendmsg(f"I:[purple]<{k}>[/purple] fetch done")
        d_ = [os.path.join(f"/home/{k}",d) for d in os.listdir(f"/home/{k}")]
        dirs = list(filter(lambda x: os.path.isdir(x),d_))
        if len(dirs) == 0:
            netc.sendmsg(f"I:[purple]<{k}>[/purple] bad fetch")
            log.write(f"[PKG] {k} : bad fetch\n")
            ret = subprocess.run(['rm','-rf',f"/home/{k}"])
            netc.sendmsg("V:V")
            k = netc.recvmsg().strip()
            continue
        
        kdir = dirs[0]
        # evaluation 
        stbl = mk_src_tbl(kdir)
        print(stbl)
        if(len(stbl) == 0):
            netc.sendmsg(f"I:[purple]<{k}>[/purple] no C file (or bad fetch)")
            log.write(f"[PKG] {k} : no C file\n")
            ret = subprocess.run(['rm','-rf',f"/home/{k}"])
            netc.sendmsg("V:V")
            k = netc.recvmsg().strip()
            continue
        # building it, 1st stop: dependencies
        tbls_f.write(f"[PKG]: {k}"+"\n"+f"{str(stbl)}"+"\n")
        netc.sendmsg(f"I:[purple]<{k}>[/purple] resolving dependencies")
        ret = subprocess.run(['apt','build-dep',k,"-y"])
        netc.sendmsg(f"I:[purple]<{k}>[/purple] dependencies resolved")
        netc.sendmsg(f"I:[purple]<{k}>[/purple] compile started")
        # compile ...
        ret = subprocess.run(['dpkg-buildpackage','-b','-uc','-us','-j8'],cwd=kdir,env=env)
        netc.sendmsg(f"I:[purple]<{k}>[/purple] compile done")
    except:
        netc.sendmsg(f"I:[purple]<{k}>[/purple] pkg error")
        log.write(f"[PKG] {k} : pkg error\n")
        ret = subprocess.run(['rm','-rf',f"/home/{k}"])
        netc.sendmsg("V:V")
        k = netc.recvmsg().strip()
        continue
    asms = glob("/tmp/*.s")
    if len(asms) == 0:
        # clear tmp
        netc.sendmsg(f"I:[purple]<{k}>[/purple] no asms")
        log.write(f"[PKG] {k} : no asms\n")
        ret = subprocess.run('rm -rf /tmp/*', shell=True)
        ret = subprocess.run(['rm','-rf',f"/home/{k}"])
        netc.sendmsg("V:V")
        k = netc.recvmsg().strip()
        continue

    netc.sendmsg(f"I:[purple]<{k}>[/purple] parse started")
    log.write(f"[PKG] {k} parse started\n")
    for asm in asms:
        smpls_c, smpls_f = handlefile(asm,stbl)
        nc = len(smpls_c) 
        nf = len(smpls_f)
        for s,t in smpls_c:
            srcf_c.write(s+"\n")
            tgtf_c.write(t+"\n")
        for s,t in smpls_f:
            srcf_f.write(s+"\n")
            tgtf_f.write(t+"\n")
        cntC += nc
        cntF += nf
    total_smpl = f"{cntC},{cntF}"
    netc.sendmsg(f"I:[purple]<{k}>[/purple] parsed C:{nc},F:{nf} funcs")
    netc.sendmsg(f"Z:{total_smpl}")
    print(f"[PKG] {k} : {cntC},{cntF}")
    log.write(f"[PKG] {k} : {cntC},{cntF}\n")
    # print(stbl)
    # clear tmp
    netc.sendmsg(f"I:[purple]<{k}>[/purple]  clearing")
    ret = subprocess.run('rm -rf /tmp/*', shell=True)
    ret = subprocess.run(['rm','-rf',f"/home/{k}"])
    netc.sendmsg(f"I:[purple]<{k}>[/purple]  cleared")
    netc.sendmsg("V:V")
    k = netc.recvmsg().strip()
    continue 
    
