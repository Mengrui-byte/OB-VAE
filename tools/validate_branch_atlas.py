"""Verify atlas selection, links, exact plotted arrays and evidence hashes."""
import hashlib,json,re,xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from tools import build_branch_atlas as b


def main():
    root=b.DOC;summary=json.loads((root/'summary.json').read_text());records=summary['records'];assert len(records)==25
    assert sum(bool(r['winners']) for r in records)==23
    counts=dict(branches=25,branches_with_recorded_result=23,branches_without_trained_result=2,architecture_svgs=0,reconstruction_cases=0,reconstruction_pngs=0,evidence_checks=0)
    for record in records:
        directory=root/record['id'];assert (directory/'README.md').exists()
        ET.parse(directory/'architecture.svg');counts['architecture_svgs']+=1
        candidates=json.loads((directory/'candidates.json').read_text())
        assert all('/pilot/' not in c['path'] for c in candidates)
        for w in record['winners']:
            c=w['candidate'];same=[v for v in candidates if v['protocol']==c['protocol']]
            assert (c['quantity'],c['price'])==min((v['quantity'],v['price']) for v in same)
            evidence=json.loads((directory/w['evidence']).read_text())
            assert b.sha(b.SOURCE/c['path'])==evidence['source_sha256']
            if c['selector'].startswith('line:'):
                line=int(c['selector'].split(':')[1]);raw=json.loads((b.SOURCE/c['path']).read_text().splitlines()[line-1])
                assert raw==evidence['record']
            else:
                raw=json.loads((b.SOURCE/c['path']).read_text());value=raw.get(c['selector']) if c['selector'] else raw
                # Normal evaluation evidence retains the wrapper; results.json stores selected entry.
                assert evidence['record']==(value if c['selector'] not in ['', 'result'] else raw)
            counts['evidence_checks']+=1
            p=w['profile']
            if p:
                a=np.load(b.SOURCE/p['source']);shown=np.load(directory/(Path(p['images'][0]).stem.removesuffix('_btc')+'_display_points.npz'))
                assert b.sha(b.SOURCE/p['source'])==p['source_sha256']
                for ri,minute in enumerate(p['minutes']):
                    assert np.array_equal(shown['truth'][ri],a['truth'][0,minute],equal_nan=True)
                    assert np.array_equal(shown['prediction'][ri],a[p['prediction_key']][0,minute],equal_nan=True)
                assert shown['truth'].shape[1:]==(2,2048,2)
                assert shown['truth'].shape==shown['prediction'].shape
                for image in p['images']:assert (directory/image).exists()
                assert len(p['images'])==3
                counts['reconstruction_cases']+=1;counts['reconstruction_pngs']+=3
        for f in directory.glob('best_*_checkpoint.json'):
            meta=json.loads(f.read_text())
            if meta['available']:assert (b.SOURCE/meta['path']).stat().st_size==meta['bytes']
    missing=[]
    for p in root.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http:','https:','#')):continue
            if not (p.parent/unquote(target.split('#')[0])).exists():missing.append((str(p),target))
    for target in re.findall(r'(?:href|src)="([^"]+)"',(root/'index.html').read_text()):
        if not target.startswith(('http:','https:','#')) and not (root/unquote(target)).exists():missing.append(('index.html',target))
    assert not missing,missing
    assert (root/'index.html').read_text().count('class="branch"')==25
    result=dict(**counts,all_relative_links_valid=True,exact_minima_per_protocol=True,plotted_data_equal_saved_predictions=True,no_synthetic_pilot_in_test_pool=True)
    (b.AUDIT/'validation.json').write_text(json.dumps(result,indent=2))
    (root/'validation.json').write_text(json.dumps(result,indent=2));print(counts)

if __name__=='__main__':main()
