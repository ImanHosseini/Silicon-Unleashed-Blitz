sat0 = open('sat0.txt','r').readlines()[0]
sat0 = sat0.split("|")
sat0 = [int(x) for x in sat0]
uns = open('ust.txt','r').readlines()
unsatZ = set()
satZ = set()
for i in range(5207):
    satZ.add(i)
for l in uns:
    if l.startswith("UNSAT CL:"):
        v = int(l.split(":")[1])
        unsatZ.add(v)
        satZ.remove(v)
for s in sat0:
    if s in unsatZ:
        print(f"{s} is UNSAT!")

for s in satZ:
    if s not in sat0:
        print(F"{s} NOT UNSAT!!")