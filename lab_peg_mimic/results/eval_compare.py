import csv, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS=[("① MimicGen","ic_mimicgen.csv","#1f77b4"),
       ("② SkillGen","ic_skillgen.csv","#2ca02c"),
       ("③ SkillGen+SART","ic_skillgen_sart.csv","#d62728")]
HOLE=(0.091,0.104); RX=(0.08,0.26); RY=(-0.12,0.12)

def load(fn):
    xs=[];ys=[];s=[]
    for r in csv.DictReader(open("results/"+fn)):
        xs.append(float(r["peg_x"]));ys.append(float(r["peg_y"]));s.append(int(float(r["success"])))
    return np.array(xs),np.array(ys),np.array(s)

data={name:load(fn)+(col,) for name,fn,col in CELLS}
sart=json.load(open("results/gen_skillgen_sart_stats.json"))

fig=plt.figure(figsize=(17,4.8))
# Panel A: DGR bar
axA=fig.add_subplot(1,4,1)
names=[c[0] for c in CELLS]; dgr=[100*data[n][2].mean() for n in names]
succ=[int(data[n][2].sum()) for n in names]; att=[len(data[n][2]) for n in names]
bars=axA.bar(range(3),dgr,color=[c[2] for c in CELLS])
for i,(d,sc,at) in enumerate(zip(dgr,succ,att)):
    axA.text(i,d+0.3,f"{d:.1f}%\n{sc}/{at}",ha="center",fontsize=9,fontweight="bold")
axA.set_xticks(range(3)); axA.set_xticklabels(["MimicGen","SkillGen","SkillGen\n+SART"],fontsize=9)
axA.set_ylabel("Data Generation Rate (%)"); axA.set_ylim(0,18)
axA.set_title("DGR  (success / attempts)",fontsize=11,fontweight="bold")
axA.grid(axis="y",alpha=.3)

# Panels B-D: IC coverage per cell
for j,(name,fn,col) in enumerate(CELLS):
    ax=fig.add_subplot(1,4,2+j)
    x,y,s,_=data[name]
    ax.add_patch(plt.Rectangle((RX[0],RY[0]),RX[1]-RX[0],RY[1]-RY[0],fill=False,ec="#999",ls="--",lw=1))
    ax.scatter(x[s==0],y[s==0],s=10,c="#ccc",alpha=.5,label="failed",edgecolors="none")
    ax.scatter(x[s==1],y[s==1],s=26,c=col,alpha=.85,label="success",edgecolors="white",linewidths=.4)
    ax.plot(*HOLE,marker="s",ms=9,c="k"); ax.annotate("socket",HOLE,fontsize=7,xytext=(3,3),textcoords="offset points")
    ax.set_xlim(0.04,0.29); ax.set_ylim(-0.16,0.16); ax.set_aspect("equal")
    ax.set_xlabel("peg x (m)"); 
    if j==0: ax.set_ylabel("peg y (m)")
    extra = f"\n20 placements × 15 approaches" if "SART" in name else ""
    ax.set_title(f"{name}  ·  {int(s.sum())}/{len(s)}{extra}",fontsize=9.5,fontweight="bold")
    if j==0: ax.legend(fontsize=7,loc="lower left")
fig.suptitle("FR3 peg-in-hole synthetic generation — 4 seeds, same spawn region, matched settings (scale 0.5, noise 0.01, interp 40)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("results/compare.png",dpi=125)
print("wrote results/compare.png")

# success grid 3x3 per cell + print
def grid(x,y,s):
    gx=np.linspace(RX[0],RX[1],4); gy=np.linspace(RY[0],RY[1],4); G=np.full((3,3),np.nan)
    for a in range(3):
        for b in range(3):
            m=(x>=gx[a])&(x<gx[a+1])&(y>=gy[b])&(y<gy[b+1])
            if m.sum(): G[b,a]=100*s[m].mean()
    return G
for name,fn,col in CELLS:
    x,y,s,_=data[name]
    print(f"\n{name}: DGR {100*s.mean():.1f}% ({int(s.sum())}/{len(s)}) | attempted xy std ({x.std():.3f},{y.std():.3f}) | success xy spread x[{x[s==1].min():.3f},{x[s==1].max():.3f}] y[{y[s==1].min():.3f},{y[s==1].max():.3f}]")
print(f"\nSART approach diversity (offset_pos_std) = {sart['offset_pos_std_m']} m ; amplifies 20 SkillGen successes -> {sart['num_success']} demos ({sart['num_success']/sart['n_sources']:.1f}x)")
print("DONE")
