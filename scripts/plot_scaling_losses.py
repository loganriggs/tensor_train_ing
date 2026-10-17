"""Regenerate the training-loss figures embedded in docs/scaling.md (CPU only)."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/figures'
OUT.mkdir(parents=True, exist_ok=True)
LRS = [('0.00015', '1.5e-4'), ('0.00045', '4.5e-4'), ('0.00135', '1.35e-3'), ('0.004', '4e-3')]
COLORS = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']
GROUPS = [
 ('scaling-width-training-loss', 'Width transfer · training loss', [
  ('12 × 512 · baseline', 'lr_transfer/w512_lr{lr}'),
  ('12 × 1024 · standard', 'lr_transfer/w1024_lr{lr}'),
  ('12 × 1024 · width scaling', 'lr_transfer/w1024_lr{lr}_wsb512')], 4),
 ('scaling-depth-training-loss', 'Depth transfer · training loss', [
  ('12 × 512 · baseline', 'lr_transfer/w512_lr{lr}'),
  ('24 × 512 · standard', 'depth_transfer/L24_lr{lr}'),
  ('24 × 512 · residual scaling', 'depth_transfer/L24_lr{lr}_drb12')], 3),
 ('scaling-joint-training-loss', 'Joint width and depth transfer · training loss', [
  ('12 × 512 · baseline', 'lr_transfer/w512_lr{lr}'),
  ('24 × 1024 · standard', 'joint_transfer/J24x1024_lr{lr}'),
  ('24 × 1024 · width + residual scaling', 'joint_transfer/J24x1024_lr{lr}_scaled')], 3)]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'svg.fonttype':'none'})
manifest = {'smoothing':'trailing 25-update token-weighted mean, partial windows at start',
            'x':'cumulative scored prediction tokens / 1e6', 'raw_lines':'faint', 'figures':{}}
for filename, title, panels, count in GROUPS:
 fig, axes = plt.subplots(1,3,figsize=(15,4.7),sharex=True,sharey=True)
 sources=[]
 for ax,(label,template) in zip(axes,panels):
  for i,(lr,lr_label) in enumerate(LRS[:count]):
   run=ROOT/'runs'/template.format(lr=lr)
   if str(run).endswith('w1024_lr0.004_wsb512'):run=Path(str(run)+'_retry')
   path=run/'metrics.jsonl';raw=path.read_bytes()
   rows=[json.loads(line) for line in raw.splitlines()]
   x=np.array([r['tokens'] for r in rows],dtype=float)
   y=np.array([r['loss'] for r in rows],dtype=float)
   weights=np.array([r['prediction_tokens'] for r in rows],dtype=float)
   assert len(rows)>0 and np.isfinite(y).all() and (np.diff(x)>0).all() and (weights>0).all()
   result=json.loads((run/'result.json').read_text())
   assert result['progress']['tokens']==int(x[-1])
   smooth=np.array([np.average(y[max(0,j-24):j+1], weights=weights[max(0,j-24):j+1]) for j in range(len(y))])
   ax.plot(x/1e6,y,color=COLORS[i],alpha=.12,linewidth=.6,rasterized=True)
   ax.plot(x/1e6,smooth,color=COLORS[i],linewidth=1.8,label='LR '+lr_label)
   sources.append({'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(raw).hexdigest(),
                   'updates':len(rows),'last_tokens':int(x[-1])})
  ax.axvline(819200/1e6,color='#777777',linestyle=':',linewidth=1)
  ax.set_title(label,fontsize=11,pad=12);ax.grid(alpha=.16)
  ax.set_xlim(0,4.8);ax.set_ylim(4.4,14.3 if count == 4 else 11.3);ax.set_xlabel('Training tokens (millions)')
 axes[0].set_ylabel('Training cross-entropy (nats / token)')
 handles,labels=axes[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=count,frameon=False)
 fig.suptitle(title,fontsize=16,y=.99,fontweight='bold')
 fig.text(.5,.02,'FineWeb pilot · one seed · uncooled runs | Bold: trailing 25-update mean; faint: raw loss | Dotted: end of 100-update warmup',ha='center',fontsize=9,color='#555555')
 fig.tight_layout(rect=(0,.16,1,.94))
 for ext in ('png','svg'):fig.savefig(OUT/(filename+'.'+ext),dpi=160,facecolor='white')
 plt.close(fig);manifest['figures'][filename]=sources
(OUT/'scaling-training-loss-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({name:len(sources) for name,sources in manifest['figures'].items()})
