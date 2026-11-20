# SPDX-License-Identifier: Apache-2.0
"""Refresh public site data atomically from the private workspace and the approved ledger."""
import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'site/public/data/training-record.json'
CLAIMS = ROOT/'site/src/data/claims.json'
LEDGER = ROOT/'lab/outreach/CLAIMS.md'
HOME = ROOT/'site/src/content/home.md'

def write_json(path, data):
    payload=(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if path.is_file() and path.read_bytes()==payload: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
        tmp.write(payload)
    Path(tmp.name).replace(path)

def approved():
    if not LEDGER.exists(): return json.loads(CLAIMS.read_text())
    text=LEDGER.read_text()
    date=re.search(r'Last verified: (\d{4}-\d{2}-\d{2})',text).group(1)
    rows={}
    for line in text.split('## Conditional')[0].splitlines():
        fields=[f.strip() for f in line.strip('|').split('|')]
        if len(fields)==4 and re.fullmatch(r'C-\d+',fields[0]):
            rows[fields[0]]=fields[1]
    # C-07 contains a disputed expansion of the treatment name and is withheld.
    rows.pop('C-07',None)
    if not rows: raise ValueError('approved claims ledger has no entries')
    return {'as_of':date,'claims':rows}

def validate(record, ledger):
    meta=record['meta']
    status_date=dt.date.fromisoformat(re.search(r'^  updated: (\d{4}-\d{2}-\d{2})',HOME.read_text(),re.M).group(1))
    if status_date < dt.date.fromisoformat(meta['exported']): raise ValueError('status date predates the newest chart data')
    scores=record['scores']
    if any(type(u) is not int or u<0 or type(v) is not int or c not in {'training','evaluation','other'} for u,v,c in scores):
        raise ValueError('invalid update, score or QA/validation category')
    if len(scores)!=meta['ai_seat_scores'] or max(u for u,v,c in scores)!=meta['latest_update']: raise ValueError('score sample coverage mismatch')
    by_update=Counter(u for u,_,_ in scores)
    audit=record['audit']
    if [[u,n] for u,n in sorted(by_update.items())]!=audit['known_update_counts']:
        raise ValueError('unknown or missing update plotted as a measured observation')
    if audit['qa_rows']!=0: raise ValueError('QA or validation row in exported population')
    for cell in record['score_mean']:
        values=[v for u,v,_ in scores if cell['lo']<=u<=cell['hi']]
        if not values or len(values)!=cell['n'] or round(sum(values)/len(values),2)!=cell['mean']:
            raise ValueError('score chart sample or aggregate mismatch')
    for window in record['windows']:
        if not window['seat_games'] or not window['policies']: raise ValueError('chart window lacks sample size')
        if 'samples' not in window: raise ValueError('metric denominator missing')
        for metric, samples in window['samples'].items():
            current=window[metric]
            for key,n in samples.items():
                value=current if key=='value' else (current or {}).get(key)
                if type(n) is not int or n<0 or (value is None)!=(n==0):
                    raise ValueError('sample coverage inconsistent with plotted value')
    home=HOME.read_text()
    for value,claim in re.findall(r"value: '([^']+)'.*?claim: (C-\d+)",home):
        wording=ledger['claims'][claim]
        if value=='0' and claim=='C-14' and 'no paid compute' in wording: continue
        if value not in wording: raise ValueError('homepage fact disagrees with approved '+claim)
    checkpoint=re.search(r'checkpoint: (\d+)',home).group(1)
    if 'update '+checkpoint+'.' not in ledger['claims']['C-05']: raise ValueError('unapproved status checkpoint')
    for p in (ROOT/'site/src/pages').glob('*.astro'):
        for heading in re.findall(r'<h[12]\b[^>]*>(.*?)</h[12]>',p.read_text(),re.S):
            if re.search(r'\d',re.sub(r'\{[^{}]*\}', '',heading)) or re.search(r'\{(?!approvedHeadlines\.)[^}]*\}',heading):
                raise ValueError(f'unreviewed numeric or dynamic headline in {p.name}')
    print(f"Validated {len(scores):,} seat scores, {len(record['windows'])} behaviour windows, {len(ledger['claims'])} dated claims.")

def refresh():
    ledger=approved()
    helper=ROOT/'gui/site-data'
    if helper.is_file() and (ROOT/'replays/games').is_dir():
        with tempfile.TemporaryDirectory() as tmp:
            dashboard=Path(tmp)/'dashboard.json'
            subprocess.run([helper,dashboard],check=True)
            record=Path(tmp)/'record.json'
            subprocess.run(['python3',ROOT/'site/scripts/export_training_record.py','--input',dashboard,'--out',record],check=True)
            values=json.loads(record.read_text())
            validate(values,ledger)
            write_json(RECORD,values)
    else:
        print('Using the reviewed aggregate distributed with this checkout.')
        validate(json.loads(RECORD.read_text()),ledger)
    write_json(CLAIMS,ledger)

def publish(confirm):
    refresh()
    paths=['site/public/data','site/src/data','site/src/content/home.md']
    subprocess.run(['git','diff','--stat','HEAD','--',*paths],cwd=ROOT,check=True)
    changed=subprocess.run(['git','diff','--quiet','HEAD','--',*paths],cwd=ROOT).returncode
    if changed not in (0,1): raise RuntimeError('could not inspect public diff')
    if not changed: print('No progress changes to commit.'); return
    branch='progress/'+dt.date.today().isoformat()
    if confirm:
        # Refuse to sweep another work item's staged changes into this commit.
        subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT,check=True)
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=branch:
            subprocess.run(['git','switch','-c',branch],cwd=ROOT,check=True)
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Refresh reviewed training charts'],cwd=ROOT,check=True)
    print(f"Review the diff, then git push -u origin {branch}")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['data','validate','publish'])
    parser.add_argument('--confirm', action='store_true')
    args=parser.parse_args()
    if args.action=='data': refresh()
    elif args.action=='publish': publish(args.confirm)
    else: validate(json.loads(RECORD.read_text()),approved())
