"""Publication plots from archived analysis only; record every plotted value."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.argoverse_hd_phase2.common import load,dump,sha
from analysis.branch_b_expanded_v1.continue_local import BASE,ROOT

OFF=['dense','original_fixed2','practical','high_ap']
LABELS=['Dense640','Original Fixed-2','Practical Fixed-2','High-AP Fixed-2']
MODES=['fifo','latest','merged'];MLABELS=['FIFO Dense','Latest Dense','Latest + Fixed-2']
COLORS=['#b44c48','#197f9c','#2b9563'];OCOLORS=['#197f9c','#9b789c','#2b9563','#df9a36']
LEDGER=[];FIGURES=[]


def get(d,keys):
    for k in keys:d=d[k]
    return d


def scalar(file,keys,scale=1):
    value=get(load(file),keys);answer=None if value is None else value*scale
    LEDGER.append(dict(kind='json',file=str(file),keys=keys,scale=scale,value=answer))
    return np.nan if answer is None else answer


def series(rows,field,scale=1):
    values=[float(r[field])*scale for r in rows]
    LEDGER.append(dict(kind='csv_mean',file=str(ROOT/'analysis/streaming.csv'),
        names=[r['name'] for r in rows],field=field,scale=scale,value=float(np.mean(values)) if values else None))
    return float(np.mean(values)) if values else np.nan


def finish(fig,name,title,interpretation,scope):
    fig.suptitle(title,fontsize=13,fontweight='bold',y=1.01)
    fig.tight_layout();dest=ROOT/'figures'/name
    fig.savefig(str(dest)+'.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(str(dest)+'.svg',bbox_inches='tight',facecolor='white')
    FIGURES.append(dict(id=name,title=title,interpretation=interpretation,scope=scope,
        png=str(dest)+'.png',svg=str(dest)+'.svg'))
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'svg.fonttype':'none','axes.unicode_minus':False})
    offline=ROOT/'analysis/offline.json';stream=ROOT/'analysis/streaming.csv'
    with stream.open() as f:rows=list(csv.DictReader(f))
    valid=[r for r in rows if r['pacing_valid']=='True']
    observed={(r['sequence'],r['trial'],int(r['rate']),r['mode']) for r in valid}
    balanced={(seq,trial) for seq,trial,_,_ in observed
        if all((seq,trial,fps,mode) in observed for fps in [15,30,60,90] for mode in MODES)}
    balanced_rows=[r for r in valid if (r['sequence'],r['trial']) in balanced]
    fig,axes=plt.subplots(1,4,figsize=(13,3.7))
    for ax,field,title in zip(axes,['AP','micro_precision','micro_recall','micro_F1'],['COCO AP (%)','Precision at .25 (%)','Recall at .25 (%)','F1 at .25 (%)']):
        values=[scalar(offline,[n,'COCO' if field=='AP' else 'fixed_conf25',field],100) for n in OFF]
        ax.bar(range(4),values,color=OCOLORS);ax.set_xticks(range(4),LABELS,rotation=35,ha='right');ax.set_title(title)
        for i,v in enumerate(values):ax.text(i,v+.3,f'{v:.2f}',ha='center',fontsize=8)
    finish(fig,'01_offline_accuracy','Four frozen offline configurations',
        'Practical and High-AP improve on Original Fixed-2 but both remain below Dense AP. Fixed-threshold precision does not substitute for recall or AP.',
        'All 3,416 images; official COCO AP; micro P/R/F1 at confidence .25 and IoU .50. Saved results reused.')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    for i,n in enumerate(OFF):
        vals=[scalar(offline,[n,'COCO',k],100) for k in ['AP_small','AP_medium','AP_large']]
        ax.bar(x+(i-1.5)*.19,vals,.19,label=LABELS[i],color=OCOLORS[i])
    ax.set_xticks(x,['Small','Medium','Large']);ax.set_ylabel('COCO AP (%)');ax.legend(fontsize=8)
    finish(fig,'02_offline_size','Object-size limitations remain',
        'Small-object AP is much lower than large-object AP for every configuration. Temporal processing does not repair the detector ceiling.',
        'Same 3,416-image population and official COCO area ranges.')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,field,label in zip(axes,['fps','cpu_ms_frame','calls'],['Decode-inclusive FPS','CPU ms / source frame','YOLO calls / 3,416 frames']):
        for i,n in enumerate(OFF):
            if field=='cpu_ms_frame' and n=='original_fixed2':continue
            xx=scalar(offline,[n,field]);yy=scalar(offline,[n,'COCO','AP'],100)
            ax.scatter(xx,yy,color=OCOLORS[i],marker='o' if n!='original_fixed2' else 's',s=65)
            ax.annotate(LABELS[i].replace(' Fixed-2',''),(xx,yy),xytext=(4,5),textcoords='offset points',fontsize=7)
        ax.set_xlabel(label);ax.set_ylabel('Official COCO AP (%)')
    finish(fig,'03_offline_tradeoffs','Accuracy, throughput, CPU and detector work',
        'Fewer calls alone do not fix the AP/cost trade-off. Practical is faster than High-AP in the matched E campaign. Original throughput is historical and unpaired; its broader-scope CPU is excluded.',
        'Same images. Dense/Practical/High-AP: five E passes. Original: one older timing campaign with different instrumentation; no connecting frontier.')
    sequences=[f'sequence_{i:02d}' for i in range(5)]
    values=np.array([[scalar(offline,[n,'per_sequence',s,'COCO','AP'],100) for s in sequences] for n in OFF])
    fig,ax=plt.subplots(figsize=(8,3.6));im=ax.imshow(values,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(5),[s.replace('sequence_','Seq ') for s in sequences]);ax.set_yticks(range(4),LABELS)
    for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>values.max()*.65 else '#16334b')
    fig.colorbar(im,ax=ax,label='Official COCO AP (%)')
    finish(fig,'04_offline_sequence','Offline performance varies by sequence',
        'The aggregate is not representative of every scene. These are reused development sequences, not unseen validation.',
        'Per-sequence official AP; class availability varies. Aggregate AP is scored jointly, not an average of these cells.')
    specs=[('05_latency',['latency_p50_ms','latency_p95_ms','latency_max_ms'],['P50 latency (ms)','P95 latency (ms)','Maximum latency (ms)'],
        'Capture deadline to output readiness','FIFO delay includes obsolete-frame waiting. Latest policies bound pending work, but their tail latency is still variable.'),
        ('06_runtime',['cpu_ms_source','active_output_fps','service_throughput_fps'],['CPU ms / source arrival','Active output FPS','Service-only throughput (FPS)'],
        'Runtime scopes answer different questions','Active FPS counts outputs inside the source-arrival window. Service throughput excludes decode, wait and drain and is not application FPS.'),
        ('07_coverage',['frame_coverage','calls','overwritten'],['Consumed / source frames','YOLO calls per sequence','Overwritten pending frames'],
        'Coverage, detector work and obsolete frames','A latest policy may consume fewer frames; the integrated mode spends some consumed steps on flow. Every source frame is offered without manual pre-dropping.'),
        ('08_resources',['peak_rss_mib','peak_pending','post_eof_drain_ms'],['Process peak RSS (MiB)','Peak pending frames','Post-EOF drain (ms)'],
        'FIFO backlog, memory and draining','Process RSS includes startup. A pending-frame bound does not bound all model, telemetry and retained-state memory.'),
        ('09_streaming_accuracy',['aligned_AP','common_AP','aligned_FP'],['Custom time-aligned AP (%)','Common-consumed-frame AP (%)','Time-aligned FP at .25'],
        'Delivered quality and per-frame accuracy','Custom held-output AP scores all source queries. Frame-aligned AP uses a conditional common-consumed population and must not be conflated with it.')]
    for name,fields,labels,title,explain in specs:
        fig,axes=plt.subplots(1,3,figsize=(12,3.8))
        for ax,field,label in zip(axes,fields,labels):
            for mode,color,mlabel in zip(MODES,COLORS,MLABELS):
                ys=[];ns=[]
                for fps in [15,30,60,90]:
                    part=[r for r in balanced_rows if r['mode']==mode and int(r['rate'])==fps]
                    ys.append(series(part,field,100 if field in ['aligned_AP','common_AP'] else 1));ns.append(len(part))
                ax.plot([15,30,60,90],ys,'o-',color=color,label=mlabel,ms=4)
                for x,y,n in zip([15,30,60,90],ys,ns):
                    if np.isfinite(y):ax.annotate(f'n={n}',(x,y),xytext=(1,5),textcoords='offset points',fontsize=6,color=color)
            if field in ['peak_rss_mib','peak_pending']:
                for p in (ROOT/'passes').glob('*/INCOMPLETE.json'):
                    error=load(p);meta=load(p.parent/'SUPERVISOR.json')
                    if field=='peak_rss_mib' and error.get('peak_rss_mib') is not None:
                        value=scalar(p,['peak_rss_mib'])
                    elif field=='peak_pending' and (error.get('buffer') or {}).get('peak_pending') is not None:
                        value=scalar(p,['buffer','peak_pending'])
                    else:continue
                    ax.scatter([meta['fps']],[value],marker='x',s=70,color='#5b2538',label='Aborted attempt')
            ax.set_xlabel('Replay rate (FPS)');ax.set_ylabel(label);ax.set_xticks([15,30,60,90])
            if field in ['latency_p50_ms','latency_p95_ms','latency_max_ms']:ax.set_yscale('log')
        axes[0].legend(fontsize=7)
        finish(fig,name,title,explain,
            f'Fixed balanced panel: {len(balanced)} scene/trial pairs from {len({s for s,t in balanced})} scenes with all three arms valid at every rate. Equal-weight run means; labels give counts. This subset is not the whole five-scene matrix. AP is mean scene AP, not pooled AP. Aborted-resource crosses are individual observed failures outside the means; all missing/invalid runs remain in tables.')
    for field,name,title in [('aligned_AP','10_sequence_aligned','Time-aligned AP across all five scenes'),('latency_p95_ms','11_sequence_latency','P95 latency across all five scenes')]:
        fig,axes=plt.subplots(1,5,figsize=(14,3.6))
        for ax,seq in zip(axes,sequences):
            for mode,col,lab in zip(MODES,COLORS,MLABELS):
                vals=[series([r for r in valid if r['mode']==mode and r['sequence']==seq and int(r['rate'])==fps],field,100 if field=='aligned_AP' else 1) for fps in [15,30,60,90]]
                ax.plot([15,30,60,90],vals,'o-',label=lab,color=col,ms=3)
            ax.set_title(seq.replace('sequence_','Seq '));ax.set_xlabel('Replay FPS');ax.set_xticks([15,30,60,90])
            if field=='latency_p95_ms':ax.set_yscale('log')
        axes[0].set_ylabel('Custom AP (%)' if field=='aligned_AP' else 'P95 milliseconds (log)');axes[0].legend(fontsize=6)
        finish(fig,name,title,'Scene-specific curves reveal consistency or reversals hidden by overall averages.',
            'Mean of complete pacing-valid trials per scene/rate/mode. No interpolation for missing observations; all trials remain in the CSV.')
    # Matched valid integrated-vs-latest comparisons; never unpaired marginal means.
    bycell={}
    for r in valid:bycell.setdefault((r['sequence'],r['rate'],r['trial']),{})[r['mode']]=r
    pairs=[]
    for cell,d in bycell.items():
        if 'latest' in d and 'merged' in d:pairs.append((cell,d['latest'],d['merged']))
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,field,title,kind in zip(axes,['aligned_AP','cpu_ms_source','latency_p95_ms'],['Time-aligned AP change (pp)','CPU reduction (%)','P95 latency reduction (%)'],['pp','relative','relative']):
        for ri,fps in enumerate([15,30,60,90]):
            values=[]
            for cell,b,c in pairs:
                if int(cell[1])!=fps:continue
                bv=series([b],field);cv=series([c],field)
                values.append(100*(cv-bv) if kind=='pp' else 100*(1-cv/bv))
            if values:
                ax.scatter(np.full(len(values),ri)+np.linspace(-.13,.13,len(values)),values,s=18,color=COLORS[2],alpha=.65)
                ax.plot([ri-.2,ri+.2],[np.mean(values)]*2,color='#17324d',lw=2)
                ax.text(ri,max(values),f'n={len(values)}',fontsize=7,ha='center',va='bottom')
        ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(4),[15,30,60,90]);ax.set_xlabel('Replay rate');ax.set_ylabel(title)
    finish(fig,'12_paired_effects','Integrated mode versus latest-frame Dense: paired effects',
        'Each point is a paired sequence/trial result. Positive CPU/latency values mean savings; AP uses percentage points. Variability is visible without a significance claim.',
        'Only cells where both B and C completed with valid arrival pacing. Horizontal segments show arithmetic means of paired effects, not confidence intervals.')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,mode,label in zip(axes,MODES,MLABELS):
        part=[r for r in rows if r['mode']==mode]
        for fps,col in zip([15,30,60,90],['#395b8c','#248f8d','#d49330','#bc4b53']):
            pp=[r for r in part if int(r['rate'])==fps]
            xx=[series([r],'common_AP',100) for r in pp];yy=[series([r],'aligned_AP',100) for r in pp]
            ax.scatter(xx,yy,s=24,label=f'{fps} FPS',color=col,alpha=.65)
        ax.set_title(label);ax.set_xlabel('Conditional frame-aligned COCO AP (%)');ax.set_ylabel('Custom replay-time-aligned AP (%)')
    axes[0].legend(fontsize=7)
    finish(fig,'13_accuracy_estimands','Frame-aligned and delivered-output AP are different estimands',
        'Each point uses different populations for its x and y coordinates. This visual describes the distinction; the diagonal is intentionally omitted and no AP difference is treated as detector degradation.',
        'All complete runs, including pacing-invalid runs for transparency; their status is identified in tables. X is a consumed intersection; Y uses all source-time queries.')
    # Execution status panel includes every intended cell; gray means incomplete.
    plan=load(ROOT/'execution_plan.json');records=load(ROOT/'analysis/results.json')['records']
    lookup={r['name']:r for r in records};fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,mode,label in zip(axes,MODES,MLABELS):
        a=np.zeros((15,4))
        for si,seq in enumerate(sequences):
            for trial in [1,2,3]:
                for fi,fps in enumerate([15,30,60,90]):
                    item=lookup[f'development_{seq}_r{trial}_{fps}_{mode}']
                    s=ROOT/'passes'/item['name'] if not item['reuse'] else Path(item['reuse'])
                    a[si*3+trial-1,fi]=0 if item['status']!='complete' else 2 if load(s/'COMPLETE.json')['valid'] else 1
        from matplotlib.colors import ListedColormap
        ax.imshow(a,aspect='auto',cmap=ListedColormap(['#a8aeb4','#e1a845','#26966a']),vmin=0,vmax=2)
        ax.set_title(label);ax.set_xticks(range(4),[15,30,60,90]);ax.set_xlabel('Replay FPS')
        ax.set_yticks(range(15),[f'S{i//3:02d} T{i%3+1}' for i in range(15)])
    finish(fig,'14_execution_status','All 180 intended streaming conditions are accounted for',
        'Green: complete and pacing-valid. Amber: complete but pacing-invalid. Gray: incomplete. Incomplete or invalid trials are never silently substituted.',
        'Five sequences x four rates x three modes x three trials; two validated reused passes count once. The earlier interrupted merged attempt is preserved separately.')
    # Queue growth is reconstructed from publication and atomic dequeue clocks.
    fig,axes=plt.subplots(1,2,figsize=(11,4));queue_ledger=[]
    for ax,fps in zip(axes,[60,90]):
        for mode,color,label in zip(MODES,COLORS,MLABELS):
            name=f'development_sequence_00_r1_{fps}_{mode}'
            item=lookup[name];out=Path(item['path'])
            if item['status']!='complete':continue
            events=load(out/'capture_events.json')
            with (out/'results.csv').open() as f: rr=list(csv.DictReader(f))
            epoch=events[0]['capture_ns']
            clocks=sorted([(int(e['published_ns']),1) for e in events]+[(int(r['selected_ns']),-1) for r in rr])
            # Replacement is also a departure from the latest pending buffer.
            # For latest, each publication sets pending=1; a dequeue sets it=0.
            pending=0;times=[];depth=[]
            for t,delta in clocks:
                if mode=='fifo':pending+=delta
                else:pending=1 if delta==1 else 0
                times.append((t-epoch)/1e9);depth.append(pending)
            assert min(depth)>=0
            assert max(depth)==load(out/'COMPLETE.json')['buffer']['peak_pending']
            ax.step(times,depth,where='post',color=color,label=label)
            queue_ledger.append(dict(name=name,out=str(out),max_pending=max(depth),last_pending=depth[-1],
                clock_events=len(clocks),mode=mode))
        ax.set_title(f'{fps} FPS: sequence_00 trial 1');ax.set_xlabel('Seconds since first source arrival');ax.set_ylabel('Pending decoded frames');ax.legend(fontsize=7)
    finish(fig,'15_queue_growth','Pending queue growth and end-of-stream drain',
        'FIFO retains obsolete arrivals and drains after EOF. Latest policies remain at zero or one pending frame. These are representative traces rather than averages.',
        'Reconstructed from original publication and atomic-selection clocks; no resampling or invented queue measurements.')
    dump(ROOT/'figures/queue_trace_audit.json',queue_ledger)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,field,label in zip(axes,['forced_full','gap_triggered_full'],['Forced fresh YOLO calls','Gap-associated forced calls']):
        for seq,col in zip(sequences,['#395b8c','#248f8d','#d49330','#bc4b53','#86649d']):
            vals=[series([r for r in valid if r['mode']=='merged' and r['sequence']==seq and int(r['rate'])==fps],field) for fps in [15,30,60,90]]
            ax.plot([15,30,60,90],vals,'o-',label=seq.replace('sequence_','Seq '),color=col)
        ax.set_xlabel('Replay FPS');ax.set_ylabel(label);ax.set_xticks([15,30,60,90])
    axes[0].legend(fontsize=7)
    finish(fig,'16_fallback_work','Safety fallbacks are part of the integrated workload',
        'The integrated mode is not guaranteed to halve calls: unsafe propagation refreshes with YOLO. Gap-associated overrides do not establish optical-flow correctness on larger gaps.',
        'Mean counts per complete pacing-valid merged run, separated by scene; raw reasons and gap distributions remain in artifacts.')
    with (ROOT/'analysis/pooled_aligned.csv').open() as f:pooled=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8,4.5))
    for mode,col,label in zip(MODES,COLORS,MLABELS):
        vals=[]
        for fps in [15,30,60,90]:
            part=[r for r in pooled if r['mode']==mode and int(r['rate'])==fps and r['pacing_valid']=='True']
            value=float(np.mean([100*float(r['AP']) for r in part])) if part else None
            LEDGER.append(dict(kind='csv_pooled',file=str(ROOT/'analysis/pooled_aligned.csv'),
                cells=[[r['mode'],r['rate'],r['trial']] for r in part],field='AP',scale=100,value=value))
            vals.append(np.nan if value is None else value)
            if value is not None:ax.annotate(f'n={len(part)}',(fps,value),xytext=(2,6),textcoords='offset points',fontsize=7,color=col)
        ax.plot([15,30,60,90],vals,'o-',color=col,label=label)
        invalid=[r for r in pooled if r['mode']==mode and r['pacing_valid']!='True']
        for r in invalid:
            LEDGER.append(dict(kind='csv_pooled',file=str(ROOT/'analysis/pooled_aligned.csv'),
                cells=[[r['mode'],r['rate'],r['trial']]],field='AP',scale=100,value=100*float(r['AP'])))
        ax.scatter([int(r['rate']) for r in invalid],[100*float(r['AP']) for r in invalid],marker='x',color=col,alpha=.45)
    ax.set_xlabel('Replay FPS');ax.set_xticks([15,30,60,90]);ax.set_ylabel('Pooled custom replay-time-aligned AP (%)');ax.legend(fontsize=8)
    finish(fig,'17_pooled_aligned','Joint held-output AP over the complete five-sequence population',
        'Joint scoring preserves all eight classes and all 3,416 source queries. It is distinct from averaging scene AP. Crosses show complete cohorts with at least one pacing-invalid stream and are excluded from the means.',
        'Each cohort combines five independently reset streams of the same arm/rate/trial. Labels count fully pacing-valid cohorts; incomplete cohorts have no pooled score.')
    conf10=ROOT/'analysis/offline_conf10.json';fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,field,title in zip(axes,['TP','FP'],['True positives at confidence .10','False positives at confidence .10']):
        values=[scalar(conf10,['methods',name,field]) for name in OFF]
        ax.bar(range(4),values,color=OCOLORS);ax.set_xticks(range(4),LABELS,rotation=30,ha='right');ax.set_title(title)
        for i,v in enumerate(values):ax.text(i,v,f'{int(v):,}',ha='center',va='bottom',fontsize=8)
    finish(fig,'18_low_confidence','High-AP weak births retain a low-confidence false-positive cost',
        'Higher High-AP AP must not be combined with Practical FP counts. At .10, weak-birth admission changes the precision/recall trade-off even when .25 precision appears similar.',
        'All 3,416 offline images; official matching at IoU .50 and confidence >=.10. High-AP also uses grid4; this is a multi-setting configuration comparison, not an isolated weak-birth ablation. Saved matching reused without inference.')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,field,title in zip(axes,['precision','recall','F1'],['Precision at .25 (%)','Recall at .25 (%)','F1 at .25 (%)']):
        for mode,col,label in zip(MODES,COLORS,MLABELS):
            vals=[]
            for fps in [15,30,60,90]:
                part=[r for r in pooled if r['mode']==mode and int(r['rate'])==fps and r['pacing_valid']=='True']
                value=float(np.mean([100*float(r[field]) for r in part])) if part else None
                LEDGER.append(dict(kind='csv_pooled',file=str(ROOT/'analysis/pooled_aligned.csv'),
                    cells=[[r['mode'],r['rate'],r['trial']] for r in part],field=field,scale=100,value=value))
                vals.append(np.nan if value is None else value)
            ax.plot([15,30,60,90],vals,'o-',color=col,label=label)
        ax.set_xlabel('Replay FPS');ax.set_xticks([15,30,60,90]);ax.set_ylabel(title)
    axes[0].legend(fontsize=7)
    finish(fig,'19_streaming_operating_points','Fixed-threshold held-output precision, recall and F1',
        'At 90 FPS the integrated mode improves the .25 operating point while reducing AP. A fixed confidence/IoU point and AP over thresholds and IoUs can rank configurations differently.',
        'Means of three pacing-valid five-sequence cohorts, each scoring all 3,416 queries; confidence >=.25, IoU .50. FIFO has no valid 90 FPS cohort and its curve stops at 60 FPS. These remain custom held-output measurements.')
    dump(ROOT/'figures/manifest.json',dict(figures=FIGURES,plotted_values=LEDGER,
        balanced_scene_trial_panel=sorted([list(p) for p in balanced]),
        source_sha256={str(p):sha(p) for p in [offline,stream,conf10,ROOT/'analysis/results.json',ROOT/'analysis/pooled_aligned.csv']},
        no_new_measurements=True))
    print('Figures generated',len(FIGURES),'plotted values',len(LEDGER))


if __name__=='__main__':main()
