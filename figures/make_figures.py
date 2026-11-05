# -*- coding: utf-8 -*-
"""Rebuild Fig1-Fig4 for the PRIA manuscript from the result files.
Run:  python make_figures.py      (writes FigN.eps, FigN.pdf, FigN.tif next to this script)
Every plotted number is asserted against the value quoted in the text."""
import os, sys, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.join(HERE,"..") if os.path.isdir(os.path.join(HERE,"..","results")) else os.path.join(HERE,"..","source","reproducibility","repo")   # repository layout or release layout
RES=os.path.join(SRC,"results"); sys.path.insert(0,os.path.join(SRC,"diagnostics"))
from grid_consistency import linear_schedule, uniform_grid, check_grid
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "axes.linewidth":0.6,"lines.linewidth":1.0,"xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.minor.width":0.5,"ytick.minor.width":0.5,
    "ps.fonttype":42,"pdf.fonttype":42,"axes.grid":False,"legend.frameon":False,"savefig.facecolor":"white","axes.facecolor":"white",
    "mathtext.fontset":"custom","mathtext.rm":"Arial","mathtext.it":"Arial:italic","mathtext.bf":"Arial:bold"})   # all lettering in Arial, incl. symbols
W,H=17.5/2.54,6.0/2.54
K="black"; G="0.45"
def fig2():
    return plt.subplots(1,2,figsize=(W,H),constrained_layout=True)
def label(ax,s): ax.text(0.0,1.02,s,transform=ax.transAxes,fontsize=8,va="bottom",ha="left")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, NullLocator
def plain_log(ax,axis,ticks):
    A=ax.yaxis if axis=="y" else ax.xaxis
    A.set_major_locator(FixedLocator(ticks)); A.set_major_formatter(FixedFormatter([("%g"%t) for t in ticks])); A.set_minor_formatter(NullFormatter())
def save(fig,name):
    for ext in ("eps","pdf"): fig.savefig(os.path.join(HERE,f"{name}.{ext}"))
    png=os.path.join(HERE,f"_{name}_600.png"); fig.savefig(png,dpi=600); plt.close(fig)
    im=Image.open(png).convert("RGB").convert("CMYK"); im.save(os.path.join(HERE,f"{name}.tif"),dpi=(600,600),compression="tiff_lzw")
    qa=os.path.join(HERE,"_qa"); os.makedirs(qa,exist_ok=True); g=Image.open(png).convert("L"); g.save(os.path.join(qa,f"{name}_grey.png")); os.remove(png)
    w_cm=im.size[0]/600*2.54; h_cm=im.size[1]/600*2.54
    assert 17.0<=w_cm<=17.55 and 5.5<=h_cm<=6.5,(name,w_cm,h_cm)
    print(f"{name}: {w_cm:.2f} x {h_cm:.2f} cm, 600 dpi, CMYK 8 bit/channel")
# ---------------- Fig. 1 (Table 1, T = 1000)
al=linear_schedule(1000,alpha_bar_T=1e-3)
N=np.unique(np.round(np.logspace(0,3,60)).astype(int))
rho=[];fac=[]
for n in N:
    r=check_grid(al,uniform_grid(1000,n)); rho.append(100*r["fraction"]); fac.append(r["realised"])
B=[1000,500,200,100,50,10]; rb=[check_grid(al,uniform_grid(1000,n)) for n in B]
assert [round(x["realised"],2) for x in rb]==[31.62,5.61,1.99,1.41,1.19,1.04]
assert [round(100*x["fraction"]) for x in rb]==[100,50,20,10,5,1]
req=rb[0]["required"]; assert round(req,2)==31.62
f,(a,b)=fig2()
a.plot(N,rho,color=K,ls="-",lw=1.0); a.plot(B,[100*x["fraction"] for x in rb],ls="none",marker="o",ms=4,mfc="white",mec=K,mew=0.8)
a.set_xscale("log"); a.set_xlabel("Retained steps $N$"); a.set_ylabel("Trajectory traversed, %"); a.set_ylim(0,105); label(a,"(a)")
b.plot(N,fac,color=K,ls="-",lw=1.0,label="realised"); b.plot(B,[x["realised"] for x in rb],ls="none",marker="o",ms=4,mfc="white",mec=K,mew=0.8)
b.axhline(req,color=G,ls="--",lw=1.0,label="required (31.62)")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("Retained steps $N$"); b.set_ylabel("Cumulative rescaling"); b.set_ylim(0.8,60); plain_log(b,"y",[1,3,10,31.62]); b.legend(loc="center left"); label(b,"(b)")
save(f,"Fig1")
# ---------------- Fig. 2 (e1d_fid.csv, T = 1000)
e=pd.read_csv(os.path.join(RES,"e1d_fid.csv")); e=e[e.T_train==1000]
f,axs=fig2()
for ax,eta,lab in [(axs[0],1.0,"(a)"),(axs[1],0.0,"(b)")]:
    I=e[(e.eta==eta)&e.form.str.startswith("I_")].set_index("nfe").fid; II=e[(e.eta==eta)&e.form.str.startswith("II_")].set_index("nfe").fid
    paired=[n for n in I.index]          # NFE at which form (I) was run
    fin=[n for n in paired if np.isfinite(I[n])]; nonfin=[n for n in paired if not np.isfinite(I[n])]
    ax.plot(paired,[II[n] for n in paired],color=K,ls="-",marker="s",ms=4,mfc=K,mec=K,lw=1.0,label="form (II), re-derived $\\varepsilon$")
    ax.plot(fin,[I[n] for n in fin],color=K,ls="--",marker="o",ms=4,mfc="white",mec=K,mew=0.8,lw=1.0,label="form (I), original $\\varepsilon$")
    top=470 if eta==0 else 170
    ax.plot(nonfin,[top]*len(nonfin),ls="none",marker="x",ms=5,mew=1.0,color=K,label="form (I) non-finite")
    ax.set_xscale("log"); ax.set_xlabel("Sampling steps (NFE)"); ax.set_ylabel("FID"); ax.set_ylim(0,top*1.12 if eta==0 else 190); label(ax,lab)
    if eta==1.0:
        assert round(I[100]-II[100],1)==61.4 and round(I[10]-II[10],1)==0.9
    else:
        assert round(I[100]-II[100],1)==321.0 and round(I[10]-II[10],1)==7.2
    assert nonfin==[200,500,1000]
    ax.legend(loc="upper left")
save(f,"Fig2")
# ---------------- Fig. 3 (e1d_gain.csv, T = 1000)
g=pd.read_csv(os.path.join(RES,"e1d_gain.csv")); g=g[g.T_train==1000].sort_values("t")
assert g[g.t>=700].g.abs().max()<0.004 and g.set_index("t").g[800]>0>g.set_index("t").g[700] and round(abs(g.set_index("t").g[10])*100,1)==36.7
f,(a,b)=fig2()
a.plot(g.t,g.g,color=K,ls="-",marker="o",ms=4,mfc="white",mec=K,mew=0.8); a.axhline(0,color=G,lw=0.6,ls=":")
a.set_yscale("symlog",linthresh=0.003,linscale=0.6); a.set_ylim(-0.6,0.008); a.yaxis.set_major_locator(FixedLocator([-0.3,-0.1,-0.03,-0.01,0,0.003])); a.yaxis.set_major_formatter(FixedFormatter(["\u22120.3","\u22120.1","\u22120.03","\u22120.01","0","0.003"])); a.yaxis.set_minor_locator(NullLocator()); a.set_xlabel("Step index $t$"); a.set_ylabel("Signed gain error $\\hat g(t)$"); a.set_xlim(1000,0); label(a,"(a)")
b.fill_between([g.t.min()-5,g.t.max()+50],0.01,0.10,facecolor="none",edgecolor=G,hatch="////",lw=0.0,label="oracle sweep, $g$ = 0.01–0.10")
b.plot(g.t,g.g.abs(),color=K,ls="-",marker="o",ms=4,mfc="white",mec=K,mew=0.8,label="$|\\hat g(t)|$")
b.plot(g.t,g.resid_frac,color=K,ls="--",marker="^",ms=4,mfc=K,mec=K,label="unexplained fraction")
b.set_yscale("log"); b.set_ylim(1e-3,8); plain_log(b,"y",[0.001,0.01,0.1,1]); b.set_xlabel("Step index $t$"); b.set_ylabel("Magnitude"); b.set_xlim(1000,0); b.legend(loc="upper left"); label(b,"(b)")
save(f,"Fig3")
# ---------------- Fig. 4 (Tables 2, 5, 6)
c=pd.read_csv(os.path.join(RES,"E1c_results_SimpleUNet.csv")); c=c[c.nfe<=50].sort_values(["T","nfe"])
d2=c.DDPM_repo.round(1).tolist(); d3=c.DDPM.round(1).tolist()
assert len(c)==12 and (min(d2),max(d2))==(383.5,535.6) and (min(d3),max(d3))==(85.7,152.0)
s=pd.read_csv(os.path.join(RES,"E1c_seeds3_SimpleUNet.csv")); sd=(round(s.sd.min(),1),round(s.sd.max(),1)); assert sd==(12.4,29.9)
ee=pd.read_csv(os.path.join(RES,"e1d_fid.csv")); ee=ee[ee.T_train==1000]
cb=tuple(round(float(ee[(ee.eta==eta)&ee.form.str.startswith("I_")&(ee.nfe==100)].fid.iloc[0]-ee[(ee.eta==eta)&ee.form.str.startswith("II_")&(ee.nfe==100)].fid.iloc[0]),1) for eta in (1.0,0.0))
assert cb==(61.4,321.0)
ss=pd.read_csv(os.path.join(RES,"E3b_samplesize.csv")); shift_cifar_len=round(float((ss.n2000-ss.n10000).mean()),2); assert shift_cifar_len==12.50
s3=pd.read_csv(os.path.join(RES,"e3a_samplesize.csv")).pivot(index="seed",columns="n",values="fid"); mn=round(float((s3[3000]-s3[10000]).mean()),2)
s2=pd.read_csv(os.path.join(RES,"e2c_fid_samplesize.csv")).pivot_table(index=["seed","solver"],columns="n",values="fid"); cs=round(float((s2[2000]-s2[10000]).mean()),2)
assert (mn,cs)==(3.27,15.56)
f,(a,b)=fig2()
x=np.arange(12); lbl=[f"{int(t)}/{int(n)}" for t,n in zip(c["T"],c.nfe)]
for i in x: a.plot([i,i],[d3[i],d2[i]],color=G,lw=0.6)
a.plot(x,d2,ls="none",marker="o",ms=4,mfc=K,mec=K,label="Eq. (2), subsampled grid")
a.plot(x,d3,ls="none",marker="o",ms=4,mfc="white",mec=K,mew=0.8,label="Eqs. (3), (4)")
a.set_yscale("log"); a.set_ylim(60,6000); plain_log(a,"y",[100,200,500,1000]); a.set_xticks(x); a.set_xticklabels(lbl,rotation=60); a.set_xlabel("$T$ / NFE"); a.set_ylabel("FID"); a.legend(loc="upper left",ncol=1,handletextpad=0.3,borderaxespad=0.3); label(a,"(a)")
items=[("class B cost, $\\eta$ = 1",cb[0],cb[0]),("class B cost, $\\eta$ = 0",cb[1],cb[1]),("seed s.d., Table 5",sd[0],sd[1]),
       ("sample-size shift, MNIST",mn,mn),("sample-size shift, CIFAR-10 length",shift_cifar_len,shift_cifar_len),("sample-size shift, CIFAR-10 samplers",cs,cs)]
for k,(nm,lo,hi) in enumerate(items):
    y=len(items)-1-k
    if lo!=hi: b.plot([lo,hi],[y,y],color=K,lw=2.0,solid_capstyle="butt"); b.plot([lo,hi],[y,y],ls="none",marker="|",ms=6,color=K)
    else: b.plot([lo],[y],ls="none",marker="D",ms=4,mfc=K if k<2 else "white",mec=K,mew=0.8)
b.set_yticks(range(len(items))); b.set_yticklabels([it[0] for it in items][::-1]); b.set_xscale("log"); b.set_xlim(1,1000); plain_log(b,"x",[1,10,100,1000]); b.set_xlabel("FID points"); b.set_ylim(-0.6,len(items)-0.4); label(b,"(b)")
save(f,"Fig4")
