"""Check every saved node card against CSV and every neighbour against parquet.

Optional QA command (requires Node.js):
    python tests/check_all_viewer_nodes.py --data data --out out

Uses the existing DOM stub, not a browser. It cannot verify pixels, browser
downloads, real keyboard focus, animation, or the team's presentation time.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd


NODE_CHECK = r"""
const fs = require('fs'), vm = require('vm');
const checker = process.argv[1], html = process.argv[2];
const expectedJson = fs.readFileSync(0, 'utf8');
const extra = `
const allExpected = new Map(expected.nodes.map(n=>[n.gid,n]));
assert.strictEqual(expected.nodes.length,allExpected.size,'Unique CSV gid');
assert.strictEqual(data.nodes.length,allExpected.size,'Viewer node count vs CSV');
assert.strictEqual(new Set(data.nodes.map(n=>n.gid)).size,allExpected.size,'Unique viewer gid');
const expectedLinks = new Map(expected.nodes.map(n=>[n.gid,[]]));
for(const edge of expected.edges){
 expectedLinks.get(edge.src).push('→ '+edge.dst);
 if(edge.src!==edge.dst)expectedLinks.get(edge.dst).push('← '+edge.src);
}
const edgeKey = e=>e.src+'|'+e.dst;
const rawEdges=new Map(expected.edges.map(e=>[edgeKey(e),e]));
assert.strictEqual(data.edges.length,rawEdges.size);
assert.strictEqual(new Set(data.edges.map(edgeKey)).size,rawEdges.size);
for(const edge of data.edges){
 const raw=rawEdges.get(edgeKey(edge));assert(raw,'Unknown directed edge '+edgeKey(edge));
 assert.strictEqual(edge.n_tx,raw.n_tx,'Edge transaction count '+edgeKey(edge));
 assert.strictEqual(Math.round(edge.sum_kzt*100),Math.round(raw.sum_kzt*100),'Edge amount vs parquet '+edgeKey(edge));
}
function checkScore(text,expectedValue,digits,context){
 assert(new RegExp('^[01],[0-9]{'+digits+'}$').test(text),'Score display precision '+context);
 const value=Number(text.replace(',','.'));
 assert(value>=0&&value<=1,'Score display range '+context);
 assert(Math.abs(value-expectedValue)<=0.5*Math.pow(10,-digits)+1e-12,'Displayed score vs CSV '+context);
}
function checkTxtAmount(text,prefix,amount,count,suffix,gid){
 const line=text.split('\\n').find(value=>value.startsWith(prefix));
 assert(line,'Missing TXT amount '+gid);
 const rendered=line.slice(prefix.length,line.indexOf(' KZT')).replace(/\\s/g,'').replace(',','.');
 // The product formats TXT amounts to whole KZT. Check the numeric error
 // bound, rather than copying its formatter or claiming fractional-KZT precision.
 assert(rendered.length>0&&Number.isFinite(Number(rendered)));
 assert(Math.abs(Number(rendered)-amount)<=0.500000001,'TXT rounded amount '+gid);
 assert(line.endsWith(suffix(count)),'TXT counterparty count '+gid);
}
let inspected=0,neighbours=0;
const allRoleLabels={coordinator:'Координатор',consolidator:'Консолидатор',distributor:'Распределитель',transit:'Транзитный',terminal:'Получатель',peripheral:'Периферийный'};
for(const n of expected.nodes){
 elements.get('gidSearch').value=n.gid;
 elements.get('searchForm').listeners.submit({preventDefault(){}});
 assert.strictEqual(api.selected,n.gid);
 assert(api.visibleNodeIds.includes(n.gid),'Selected node must be visible '+n.gid);
 assert.strictEqual(card.childNodes.find(e=>e.className==='gid-label').textContent,n.gid);
 const payload=data.nodes.find(item=>item.gid===n.gid);
 assert.strictEqual(payload.role,n.role,'Role vs CSV '+n.gid);
 assert.strictEqual(payload.cluster_id,n.cluster_id,'Cluster vs CSV '+n.gid);
 assert.strictEqual(payload.role_score,n.role_score,'Role score vs CSV '+n.gid);
 assert.strictEqual(payload.priority_score,n.priority_score,'Priority score vs CSV '+n.gid);
 assert.strictEqual(elements.get('cardTag').textContent,'Кластер '+n.cluster_id,'Visible cluster '+n.gid);
 checkScore(metric('Приоритет проверки'),n.priority_score,4,n.gid+' priority');
 checkScore(metric('Сила правила роли'),n.role_score,2,n.gid+' role');
 assert.strictEqual(card.childNodes.find(e=>e.className==='badge').textContent,allRoleLabels[n.role],'Visible role '+n.gid);
 assert.strictEqual(metric('Плательщики → получатели'),nf.format(n.in_deg)+' → '+nf.format(n.out_deg));
 assert.strictEqual(metric('Переводы: вход / выход'),nf.format(n.in_tx)+' / '+nf.format(n.out_tx));
 assert.strictEqual(card.childNodes.find(e=>e.className==='evidence').textContent,n.evidence,'Exact evidence '+n.gid);
 const allLimits=descendants(card).find(e=>e.className==='limits');
 assert(allLimits.childNodes.some(e=>e.textContent==='Следующий запрос: '+n.next_request),'Exact next request '+n.gid);
 const buttons=descendants(card).filter(e=>e.tagName==='button'&&e.className==='neighbor');
 const displayedLinks=buttons.map(e=>e.childNodes[0].textContent).sort();
 assert.deepStrictEqual(displayedLinks,[...expectedLinks.get(n.gid)].sort(),'Incoming/outgoing links '+n.gid);
 neighbours+=buttons.length;
 descendants(card).find(e=>e.tagName==='button'&&e.textContent==='Скачать карточку ↓').click();
 assert(downloaded.includes('Карточка клиента '+n.gid));
 const txtLines=downloaded.split('\\n');
 assert(txtLines.includes('Роль: '+allRoleLabels[n.role]),'TXT role '+n.gid);
 assert(txtLines.some(line=>line.startsWith('Кластер '+n.cluster_id+': ')),'TXT cluster '+n.gid);
 checkScore(txtLines.find(line=>line.startsWith('Приоритет: '))?.slice('Приоритет: '.length)||'',n.priority_score,4,n.gid+' TXT priority');
 checkScore(txtLines.find(line=>line.startsWith('Сила правила: '))?.slice('Сила правила: '.length)||'',n.role_score,2,n.gid+' TXT role');
 assert(downloaded.includes('\\n'+n.evidence+'\\n'),'TXT evidence '+n.gid);
 assert(downloaded.includes('\\nСледующий запрос: '+n.next_request+'\\n'),'TXT next request '+n.gid);
 checkTxtAmount(downloaded,'Вход: ',n.in_kzt,n.in_deg,count=>' KZT от '+count+' плательщиков',n.gid);
 checkTxtAmount(downloaded,'Выход: ',n.out_kzt,n.out_deg,count=>' KZT к '+count+' получателям',n.gid);
 inspected++;
}
assert.strictEqual(neighbours,expected.edges.reduce((count,edge)=>count+(edge.src===edge.dst?1:2),0));
console.log(JSON.stringify({status:'PASS',nodes_checked:inspected,edges_checked:expected.edges.length,neighbour_entries_checked:neighbours,txt_payloads_checked:inspected,mode:'DOM stub; not a visual browser check'}));
`;
// Parse fixtures in the same VM realm, so deep comparisons test data rather
// than cross-context Array prototypes.
vm.runInNewContext('const expected=JSON.parse(expectedJson);\n'+fs.readFileSync(checker,'utf8')+extra,
 {require,console,process:{argv:['node',checker,html]},expectedJson}, {timeout:120000});
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('out'))
    args = parser.parse_args()
    node = shutil.which('node')
    if not node:
        parser.error('Node.js is required only for this optional DOM QA command')
    with (args.out / 'nodes_roles.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in ('in_deg', 'out_deg', 'in_tx', 'out_tx'):
            row[name] = int(row[name])
        for name in ('in_kzt', 'out_kzt', 'role_score', 'priority_score'):
            row[name] = float(row[name])
        if not row['evidence'] or not row['next_request']:
            raise ValueError('Empty evidence or next_request in CSV')
    source_nodes = pd.read_parquet(args.data / 'nodes.parquet')
    if {row['gid'] for row in rows} != {str(gid) for gid in source_nodes.gid}:
        raise ValueError('CSV node coverage differs from parquet')
    edges = [{'src': str(row.src), 'dst': str(row.dst),
              'sum_kzt': float(row.sum_kzt), 'n_tx': int(row.n_tx)}
             for row in pd.read_parquet(args.data / 'edges.parquet').itertuples()]
    checker = Path(__file__).resolve().parents[1] / 'scripts/check_viewer.cjs'
    result = subprocess.run(
        [node, '-e', NODE_CHECK, str(checker), str((args.out / 'network.html').resolve())],
        input=json.dumps({'nodes': rows, 'edges': edges}, ensure_ascii=False),
        encoding='utf-8', capture_output=True, timeout=150,
    )
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
