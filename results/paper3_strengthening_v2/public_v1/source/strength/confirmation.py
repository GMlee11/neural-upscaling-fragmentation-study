"""Finite matched-geometry generator and independently balanced E002 schedule."""
import random

from .cached import POLICIES

MASK_SEED=20260925031
ORDER_SEED=20260925032
FAMILIES=(('sparse_all_rows',(0,1,2,3,4,5),2),
          ('sparse_three_rows',(0,2,5),4),
          ('dense_all_rows',(0,1,2,3,4,5),6),
          ('dense_four_rows',(0,1,3,5),9))


def matched_masks(excluded,seed=MASK_SEED):
    rng=random.Random(seed);used={tuple(rows) for rows in excluded};masks=[]
    for family,active,count in FAMILIES:
        for replica in range(2):
            equal_catalog=family=='dense_all_rows' and replica==0
            for attempt in range(1000):
                compact=['0000000000']*6;fragmented=['0000000000']*6
                for row in active:
                    left=rng.randrange(11-count)
                    compact[row]=''.join('1' if left<=c<left+count else '0' for c in range(10))
                    if equal_catalog:
                        origin=rng.randrange(3)
                        columns=[origin,origin+7]+rng.sample(list(range(origin+1,origin+7)),4)
                    else:
                        for _ in range(1000):
                            columns=sorted(rng.sample(range(10),count))
                            if max(columns)-min(columns)+1>=min(10,count+2):break
                        else:raise ValueError('fragmentation geometry budget exhausted')
                    fragmented[row]=''.join('1' if c in columns else '0' for c in range(10))
                a,b=tuple(compact),tuple(fragmented)
                if a not in used and b not in used and a!=b:break
            else:raise ValueError('unique matched-mask budget exhausted; no seed change')
            pair=f'E002_{family}_{replica}'
            for treatment,value in (('compact',a),('fragmented',b)):
                used.add(value)
                masks.append({'mask_id':pair+'_'+treatment,'pair_id':pair,'family':family,
                    'replicate':replica,'member':treatment,'rows':value,'selected_tiles':len(active)*count,
                    'active_rows':list(active),'per_row_selected_counts':[row.count('1') for row in value],
                    'equal_catalog_width_control':equal_catalog,'generation_attempt':attempt+1,
                    'split':'reserved_matched_geometry'})
    return masks


def schedule(masks,seed=ORDER_SEED):
    rng=random.Random(seed);by_pair={}
    for mask in masks:by_pair.setdefault(mask['pair_id'],[]).append(mask)
    if len(masks)!=16 or len(by_pair)!=8 or any(len(pair)!=2 for pair in by_pair.values()):
        raise ValueError('exactly eight matched pairs required')
    output=[]
    for session in range(2):
        cell_order=('width24','fp16') if session==0 else ('fp16','width24')
        for cell in cell_order:
            initial={(mask['mask_id'],policy):rng.randrange(2) for mask in masks for policy in POLICIES}
            mask_initial={pair:rng.randrange(2) for pair in by_pair}
            for repeat in range(2):
                pairs=sorted(by_pair);rng.shuffle(pairs)
                for pair in pairs:
                    members=sorted(by_pair[pair],key=lambda m:m['member'])
                    if mask_initial[pair]^repeat:members.reverse()
                    for mask in members:
                        blocks=list(POLICIES)+['frozen_choice'];rng.shuffle(blocks)
                        for policy in blocks:
                            if policy=='frozen_choice':modes=('chosen',)
                            else:
                                modes=('whole','selected')
                                if initial[mask['mask_id'],policy]^repeat:modes=tuple(reversed(modes))
                            for mode in modes:
                                output.append({'order':len(output),'session':session,'cell':cell,'repeat':repeat,
                                    'mask_id':mask['mask_id'],'pair_id':pair,'policy':policy,'reconstruction':mode,
                                    'configuration':'frozen_choice' if policy=='frozen_choice' else policy+':'+mode,
                                    'rows':list(mask['rows']),'kind':'evaluation','split':'reserved_matched_geometry'})
    if len(output)!=1664:raise ValueError('changed finite schedule')
    return output
