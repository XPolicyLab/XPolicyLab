import json,glob,os,sys,time,collections
rs=sys.argv[1]
bad=collections.Counter(); tot=collections.Counter()
for f in glob.glob(rs+"/runs/bridge/*/*_ep*/result.json"):
    if time.time()-os.path.getmtime(f)>40*3600: continue
    try: r=json.load(open(f))
    except Exception: continue
    key=(r.get("task"),r.get("seed"))
    tot[key]+=1
    if not r.get("commands"): bad[key]+=1
for k in sorted(tot, key=str):
    if bad[k]: print(f"  INVALID {k[0]} seed {k[1]}: {bad[k]} of {tot[k]} episodes had zero agent commands")
print("episodes checked:",sum(tot.values()),"| with zero commands:",sum(bad.values()))
