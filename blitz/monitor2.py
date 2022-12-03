from logging import log
from re import escape
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table, Row
from rich.text import Text
import time
import datetime
from struct import pack
import os 

import socket
from threading import Thread, Lock, Timer
from collections import deque

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432+1        # Port to listen on (non-privileged ports are > 1023)
TBN = 22
TBA = 8
CHKS_F = 20
ALERT_TH = 40

status_tbl = dict()
subnms = dict()
submx = dict()
smplcnt = dict()
tot = [0,0]
lastt = dict()
msgq = deque()
alq = deque()
lock = Lock()
lockS = Lock()
lockT = Lock()
lockZ = Lock()
finished = set()
pkg_list = []
pkg_orig = 0
import apt
import apt.package

def get_pkg_list():
    cache = apt.Cache()
    ks = cache.keys()
    return ks

def checkstatus():
    t = datetime.datetime.now()
    with lockT:
        for k,v in lastt.items():
            if k not in finished:
                dt:datetime.timedelta = t-v
                if( dt.seconds/60 > ALERT_TH):
                    mins = dt.seconds//60
                    t = datetime.datetime.now()
                    tstr = t.strftime('%m-%d %H:%M:%S')
                    m = f"[red] <{k}> {tstr} unresponsive for {mins} (m)"
                    pushalrt(m)
    Timer(60*CHKS_F,checkstatus).start()

checkstatus()

def pushalrt(msg):
    with lock:
        alq.pop()
        alq.appendleft(msg)

def pushmsg(msg):
    with lock:
        msgq.pop()
        msgq.appendleft(msg)

for _ in range(TBN-2):
    msgq.append("")

for _ in range(TBA-2):
    alq.append("")

def recvstr(sock,n):
    data = bytearray()
    while len(data) < n:
        chnk = sock.recv(n-len(data))
        if not chnk:
            return None
        data.extend(chnk)
    return data.decode('ascii')

class Handler(Thread):
    def __init__(self,ip,port,conn): 
        global subnms, smplcnt, submx
        Thread.__init__(self) 
        self.ip = ip 
        self.port = port 
        self.conn = conn
        # msg = "[black]\[+][/black] New connection " + ip + ":" + str(port)
        # pushmsg(msg)
        self.logf = None
        msg0 = self.recvmsg(zero=True)
        # H:C0:2210
        _, self.name = msg0.split(":")
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")
        self.logf = open(f"./logs/{self.name}.log","a")
        with lockS:
            subnms[self.name] = 0
            submx[self.name] = 0
            smplcnt[self.name] = (0,0)
            status_tbl[self.name] = 'FETCH'
        msg = f"[green][NET][black] new connection ({ip}:{port}) [green]<{self.name}>"
        pushmsg(msg)

    def sendmsg(self,msg):
        encd = msg.encode('ascii')
        n = len(encd)
        hd = pack('=i',n)
        try:
            self.conn.sendall(hd)
            self.conn.sendall(encd)
        except:
            print("[BLITZ] connection error")
            return

    def recvmsg(self,zero=False):
        global lstt
        data = self.conn.recv(4) 
        l = int.from_bytes(data,'little')
        if l == 0:
            return None
        msg = recvstr(self.conn,l)
        t = datetime.datetime.now()
        if not zero:
            with lockT:
                lastt[self.name] = t
        if self.logf != None:
            self.logf.write(f"{t.strftime('%m-%d %H:%M:%S')}; {msg}"+"\n")
            self.logf.flush()
        return msg

    def handlemsg(self,msg):
        global tot, subnms, smplcnt, pkg_list, submx
        mtype,txt = msg.split(":",maxsplit=1)
        # I:TXT
        # I:[pkgname] build started
        if mtype == "V":
            with lockZ:
                if len(pkg_list) > 0:
                    k = pkg_list.pop()
                    self.sendmsg(f"{k}")
                else:
                    self.sendmsg(f"ZXYXZ")
                    finished.add(self.name)
                    status_tbl[self.name] = " DONE"
            return
        if mtype == "I":
            t = datetime.datetime.now()
            tstr = t.strftime('%m-%d %H:%M:%S')
            m = f"[green]<{self.name}> {tstr}[/green] {txt}"
            pushmsg(m)
            if "parsed" in txt:
                with lockS:
                    subnms[self.name] = subnms[self.name] + 1
                return
            if "fetch started" in txt:
                with lockS:
                    submx[self.name] = submx[self.name] + 1
                    status_tbl[self.name] = "FETCH"
                return
            if "resolving" in txt:
                with lockS:
                    status_tbl[self.name] = "DEPEN"
                return
            if "compile started" in txt:
                with lockS:
                    status_tbl[self.name] = "CMPIL"
                return
            if "parse started" in txt:
                with lockS:
                    status_tbl[self.name] = "PARSE"
                return
            if "clearing" in txt:
                with lockS:
                    status_tbl[self.name] = "CLEAR"
                return
            return
        if mtype == "Z":
            nC, nF = [int(x.strip()) for x in txt.split(",",maxsplit=1)]
            with lockS:
                smplcnt[self.name] = (nC,nF)
                vs = smplcnt.values()
            tot_ = [0,0]
            for vC,vF in vs:
                tot_ = [tot_[0]+vC,tot_[1]+vF]
            with lockS:
                tot = tot_.copy()
            return

    def run(self): 
        DCN = 20
        dc = 0
        while dc<DCN : 
            msg = self.recvmsg()
            if msg is None:
                dc += 1
            else:
                dc = 0
                self.handlemsg(msg)
        # print("conn closed")
        t = datetime.datetime.now()
        tstr = t.strftime('%m-%d %H:%M:%S')
        pushalrt(f"[red] {tstr} <{self.name}> disconnected")
        self.logf.write(f"{tstr}; <{self.name}> disconnected\n")
        finished.add(self.name)
        status_tbl[self.name] = " DEAD"
        self.logf.close()
        self.conn.close()

class Monit(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.threads = []
        self.tcpServer = None

    def run(self) -> None:
        self.tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        self.tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        self.tcpServer.bind((HOST, PORT)) 
        self.threads = [] 
        try:
            while True: 
                self.tcpServer.listen(10) 
                # print("BLITZ monitor : waiting for conns")
                (conn, (ip,port)) = self.tcpServer.accept() 
                newthread = Handler(ip,port,conn) 
                newthread.start() 
                self.threads.append(newthread) 
        except:
            # print("EXCEPT")
            return


def make_layout() -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="updates",size=TBN),
        Layout(name="status",size=20),
        Layout(name="alerts",size=TBA)
    )
    return layout
from collections import deque
import time


t0 = datetime.datetime.now()
# submsg['C0'] = '[12/1200]'
# subnums['C0'] = 12

class Status:
    def __init__(self,console,layout:Layout) -> None:
        self.console = console
        self.layout = layout
        pass

    def __rich__(self) -> Panel:
        grid = Table.grid(expand = False)
        grid.add_column(justify="left", ratio = 1)
        grid.add_column()
        subt = dict()
        with lockS:
            for k,v in subnms.items():
                subt[k] = f"{v}/{submx[k]}"
        stat = []
        for k,v in status_tbl.items():
            perc = f"{subt[k]}"
            stat.append('{0: <23}'.format(f"<{k}>:{v} {perc}"))
        # for i in range(64):
        #     stat.append(f"<c{i}>:FETCH")
        w = self.console.width - 4
        left = w
        while len(stat)>0:
            left = w
            txt = ""
            while len(stat)>0 and left>(len(stat[-1])+3):
                s = stat.pop()
                sadd = f" | {s}"
                left -= len(sadd)
                txt += sadd
            grid.add_row(txt)
        # grid.add_row(stat)
        layout["status"].size = grid.row_count+2
        # layout["status"].height = grid.row_count+2
        grid.title_justify = "center"
        return Panel(grid,style="white on green",title=f"[blue]STATUS")

class AlertBar:
    def __init__(self,console,layout) -> None:
        self.console = console
        self.layout = layout
        pass

    def __rich__(self) -> Panel:
        grid = Table.grid(expand = False)
        grid.add_column(justify="left", ratio = 1)
        grid.add_column()
        for v in alq:
            grid.add_row(v)
        grid.title_justify = "center"
        return Panel(grid,style="white on blue",title=f"[red]ALERTS")

class TopBar:
    def __init__(self) -> None:
        # self.q = deque()
        # for _ in range(TBN-2):
        #     self.q.append("")
        pass

    def __rich__(self) -> Panel:
        global subnms, smplcnt
        grid = Table.grid(expand = False)
        grid.add_column(justify="left", ratio = 1)
        grid.add_column()
        with lock:
            for v in msgq:
                grid.add_row(v)
        grid.title_justify = "center"
        dt = datetime.datetime.now()-t0
        subt = f"| SAMPLS: C:{tot[0]} F:{tot[1]} | PKG: {pkg_orig - len(pkg_list)} / {pkg_orig} |"
        return Panel(grid,style="white on blue",title=f"[red]BLITZv2.0[/red] | RUNTIME: {dt.seconds/60:.2f} (m)",subtitle=subt,subtitle_align="left")

    # def add(self,x):
    #     self.q.pop()
    #     self.q.appendleft(x)

console = Console()
layout = make_layout()
layout.visible = True
tb = TopBar()
lb = AlertBar(console,layout)
stt = Status(console,layout)
layout["updates"].update(tb)
layout["alerts"].update(lb)
layout["status"].update(stt)

from rich.live import Live
from time import sleep
pkg_list = get_pkg_list()
pkg_orig = len(pkg_list)
monit = Monit()
monit.start()

live = Live(layout, refresh_per_second=10, screen=True)
live.start()
# import random
# while(True):
#     # tb.add(f"AAAAAAAAAAAAAAAAAA {random.randint(1,199)}")
#     pushmsg(f"AAAAAAAAAA {random.randint(1,199)}")
#     # tb.grid.add_row(f"AA {random.randint(1,199)}")
#     time.sleep(2)
#     continue
