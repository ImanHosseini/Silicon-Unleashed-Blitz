import os
import re
import subprocess
from pathlib import Path
import os

########################
# FORTRAN PARSING START
########################

def fnmr(rst):
    x = rst.strip()
    if "(" in x:
        x = x.split("(")[0]
    return x.strip()

def f_getsfuns(slines):
    slit_rgx = r"""(["'])(.+?)\1"""
    sfuns_by_line = dict()
    code = ""
    sline = -1
    i = 0
    typ = None
    fnm = None
    while i < len(slines):
        sl = slines[i].strip()
        sl = sl.lower()
        sl = re.sub(slit_rgx,"\"STR\"",sl)
        if "!" in sl:
            sl = sl.split("!")[0]
            if len(sl)<2:
                i = i + 1
                continue
        if typ is None:
            if "function" in sl:
                sline = i + 1
                typ = "function"
                fnm = fnmr(sl.split(typ)[1])
                code += sl
                i = i + 1
                continue
            if "program" in sl:
                sline = i + 1
                typ = "program"
                fnm = fnmr(sl.split(typ)[1])
                code += sl
                i = i + 1
                continue
            if "subroutine" in sl:
                sline = i + 1
                typ = "subroutine"
                fnm = fnmr(sl.split(typ)[1])
                code += sl
                i = i + 1
                continue
        else:
            if sl.strip() == "contains":
                code += " # "+f"end {typ} {fnm}"
                sfuns_by_line[sline] = code
                code = ""
                typ = None
                fnm = None
                i = i + 1 
                continue
            if f"end {typ}" in sl:
                code += " # " + sl
                sfuns_by_line[sline] = code
                code = ""
                typ = None
                fnm = None
                i = i + 1 
                continue
            if sl.strip() == "end":
                code += " # " + sl
                sfuns_by_line[sline] = code
                code = ""
                typ = None
                fnm = None
                i = i + 1 
                continue
            if len(sl.strip())<2:
                i = i +1 
                continue
            code += " # " + sl
            i = i + 1 
            continue
        i = i + 1
    return sfuns_by_line

########################
# FORTRAN PARSING END
########################

def respath(pth,ftbl):
    hd, tl = os.path.split(pth)
    if pth.endswith((".f",".f90",".f95",".f03")):
        if tl in ftbl:
            # print("HIT: "+pth)
            return ftbl[tl]
        # print("CANT FIND: "+pth)
        return None
    # NOT C!
    return None

    # for i in range(5):
    #     try:
    #         xd = str(Path(file).parents[i])
    #         xdj = os.path.join(xd,pth)
    #         if os.path.isfile(xdj):
    #             return xdj
    #     except:
    #         print(f"Cant Find: {file}")
    #         return "X"
    # # cant file the file
    # print(f"NCF: {file}")
    # return "X"

# fill up the holes
def handlefile(file,table_of_srcs = dict()):
    try:
        asmlines = open(file,"r").readlines()
    except:
        return [],[]
    i = 0 
    funcs = dict()
    stmts = ""
    fno = -1
    faddr = None
    lines = []
    fls = dict()
    ftbl = dict()
    while i < len(asmlines):
        l = asmlines[i]
        if (faddr is not None) and l.startswith("\t.file"):
            fil_line = l
            if not (fil_line.split()[1].isdecimal()):
                i = i + 1
                continue
            sfilepath = fil_line.split('"')[1]
            ftbl[int(fil_line.split()[1])] = respath(sfilepath,table_of_srcs)
            i = i + 1
            continue
        if (faddr is None) and l.startswith("\t.file"):
            fil_line = l
            if len(fil_line.split())<3:
                i = i + 1 
                continue
            sfilepath = fil_line.split('"')[1]
            if len(sfilepath) > 2:
                i = i + 1 
                continue
            ftbl[int(fil_line.split()[1])] = respath(sfilepath,table_of_srcs)
            i = i + 1 
            continue
        if l.startswith("\t.cfi_startproc"):
            # go.algo.E22:
            # .LFB0:
            # 	.file 1 "/hdd/discovery/morego/d0/Project-Euler-Go/algo/22.go"
            # 	.loc 1 19 1
            # 	.cfi_startproc
            # if "__entry:" in asmlines[i-1]:
            if i<5:
                i = i + 1
                continue
            name_line = asmlines[i-2]
            if "file" in asmlines[i-2]:
                name_line = asmlines[i-3]
            loc_line = asmlines[i-1]
            if ("loc" not in loc_line) or (":" not in name_line):
                i = i + 1
                continue
            fno = int(loc_line.split()[1])
            if fno in ftbl.keys():
                sfilepath = ftbl[fno]
            else:
                # TODO: Are we sure it's not at i-3?
                fil_line = asmlines[i-2]
                if ".file" not in fil_line:
                    i = i + 1
                    continue
                else:
                    sfilepath = fil_line.split('"')[1]
                    sfilepath = respath(sfilepath,table_of_srcs)
                    ftbl[fno] = sfilepath
            sline = int(loc_line.split()[2])
            if sfilepath == None:
                i = i + 1
                continue
            if not os.path.exists(sfilepath):
                i = i + 1
                continue
            stmts = ""
            # if asmlines[i-3].startswith("."):
            #     stmts += asmlines[i-3].strip()
            faddr = (sfilepath,sline,name_line.split(":")[0])
            if sfilepath != "X":
                lines.append(sline)
            i = i + 1
            continue
        if (faddr is not None) and l.startswith("\t.cfi_endproc"):
            funcs[faddr] = stmts
            fls[faddr] = lines.copy()
            lines.clear()
            faddr = None
            stmts = ""
            i = i + 1 
            continue
        if faddr is not None:
            if l.startswith("\t.loc"):
                if int(l.split()[1]) != fno:
                    # this loc count's to a function somewhere else!
                    # probably it's a call: 
                    # call calBig_int__sys_big_int_of_string_blahblah
                    # skip it!
                    i = i + 1
                    continue
                fno = int(l.split()[1])
                line = int(l.split()[2])
                if fno not in ftbl:
                    print("Z")
                if ftbl[fno] != "X":
                    lines.append(line)
                i = i + 1
                continue
            if not l.startswith("\t."):
                l = l.replace("\t"," ")
                # KEEP <#> newline betwix instructions
                stmts += " # "+l.strip()
            i = i + 1
            continue
        i = i + 1
    # dedup lines
    for k in fls.keys():
        fls[k] = list(dict.fromkeys(fls[k]))
    # <source file path, sfuns_by_line>
    srclines = dict()
    smpls_c = []
    smpls_f = []
    # info = []
    slit_rgx = r'''(?x)   # verbose mode
    (?<!\\)    # not preceded by a backslash
    "          # a literal double-quote
    .*?        # 1-or-more characters
    (?<!\\)    # not preceded by a backslash
    "          # a literal double-quote
    '''
    stbl = dict()
    for k,v in funcs.items():
        spath, sline, fname = k
        if k not in fls:
            continue
        lines = sorted(fls[k])
        # lines = fls[k]
        if len(lines)<1:
            continue
        if spath not in srclines:
            if not os.path.isfile(spath):
                continue
            try:
                srclines[spath] = open(spath,"r").readlines()
            except:
                continue
        src = srclines[spath] 
        if spath.endswith((".c",".h")):
            continue
            # FILL-UP the lines
            lines = list(range(min(lines),max(lines)+1))

            # fixup prototypes upto 2 line!
            sl = lines[0]-1
            try:
                if src[sl].strip().startswith("{"):
                    prel = src[sl-1]
                    if sl>0:
                        # add that line
                        lines.insert(0,sl)
                    if "(" not in prel:
                        if (sl-1)>0:
                            lines.insert(0,sl-1)
            except:
                # shouldn't have happened!
                return [],[]
            # es geht los!
            tgt_seq = []
            INCOMMENT = False
            for lno in lines:
                try:
                    l = src[lno-1]
                except:
                    print("LNO err:")
                    print(file)
                    print(lno-1)
                    # raise Exception("Where is lno?")
                    continue
                aux = l.strip()
                if INCOMMENT:
                    if "*/" in aux:
                        INCOMMENT = False
                        continue
                    else:
                        continue
                # skip line comments
                if aux.startswith("//"):
                    continue
                if "//" in aux:
                    aux = aux.split("//")[0]
                if "/*" in aux:
                    if "*/" in aux:
                        continue
                    else:
                        INCOMMENT = True
                        continue
                if len(aux)<1:
                    continue
                # Remove comments at the end of line if any
                if "(*" in aux:
                    aux = aux.split("(*")[0].strip()
                ## HANDLE STR LITERALS
                aux = re.sub(slit_rgx, "\"STR\"",aux)
                tgt_seq.append(aux.strip())
            # tgt_seq = sfun_by_line_tbl[spath][sline]
            if len(tgt_seq)<1:
                continue
            tgt_seq = " # ".join(tgt_seq)
            smpls_c.append((v.strip(),tgt_seq))
            # info.append((spath,sline,fname))
        elif spath.endswith((".f90",".f95",".f03",".f")):
            if spath not in stbl:
                stbl[spath] = f_getsfuns(src)
            sfuns = stbl[spath]
            tgt_seq = None
            for l in lines:
                if l in sfuns:
                    tgt_seq = sfuns[l]
                    break
            if tgt_seq is not None:
                # FORTRAN is case-insensitive
                tgt_seq = tgt_seq.lower()
                smpls_f.append((v,tgt_seq))
    # print(f"S:{len(smpls)}")
    return smpls_c, smpls_f


# handlefile("./data/d0/Collections-C/build/src/cc_array.s")
# handlefile("./data/d1/leetcode/0089_gray_code/gray_code.s")

# handlefile("./d14/99-problems-ocaml/91-100/97.s")
# s,v = handlefile("sieve.s")
# for x in s:
#     print(x[0])
#     print(x[1])
# print(dwarfer("a.out"))