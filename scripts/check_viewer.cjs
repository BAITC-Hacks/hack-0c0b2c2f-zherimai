// DOM regression checks for generated HTML; not a visual browser test.
// Run after python run.py: node scripts/check_viewer.cjs [path/to/network.html]
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2] || 'out/network.html','utf8');
const data=JSON.parse(html.match(/<script id="graphData" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Element{
 constructor(tag='div'){this.tagName=tag;this.childNodes=[];this.style={};this.value='';this.listeners={};const classes=new Set();this.classList={toggle(name,force){const on=force===undefined?!classes.has(name):force;if(on)classes.add(name);else classes.delete(name);return on},contains:name=>classes.has(name)};this.focusCalls=[];this.scrollCalls=[];this._text='';}
 set textContent(s){this._text=String(s);this.childNodes=[];}
 get textContent(){return this._text+this.childNodes.map(x=>x.textContent).join(' ');}
 append(...xs){this.childNodes.push(...xs);}
 replaceChildren(...xs){this._text='';this.childNodes=xs;}
 addEventListener(name,fn){this.listeners[name]=fn;}
 setAttribute(){} getBoundingClientRect(){return{width:1000,height:510,left:0,top:0};}
 getContext(){return new Proxy({},{get:()=>()=>{}});}
 click(){this.listeners.click?.({preventDefault(){}});}
 focus(options){this.focusCalls.push(options)} scrollIntoView(options){this.scrollCalls.push(options)}
 setPointerCapture(){} releasePointerCapture(){}
}
const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
elements.get('graphData').textContent=JSON.stringify(data);
let downloaded,viewportWidth=1200;
const env={document:{getElementById:id=>{assert(elements.has(id),id);return elements.get(id)},createElement:tag=>new Element(tag),createTextNode:s=>({textContent:s})},window:{devicePixelRatio:1,matchMedia:query=>({matches:query==='(max-width:850px)'&&viewportWidth<=850})},location:{hash:''},requestAnimationFrame:()=>1,ResizeObserver:class{observe(){}},Blob:class{constructor(parts){this.parts=parts}},URL:{createObjectURL:b=>{downloaded=b.parts.join('');return'blob:checked'},revokeObjectURL(){}},setTimeout:()=>1};
vm.runInNewContext(script,env,{timeout:10000});
const api=env.window.DALAAI,card=elements.get('nodeCard');
const nodes=[...data.nodes].sort((a,b)=>b.priority_score-a.priority_score);
assert.strictEqual(api.selected,null,'Initial overview must not select a client');
assert.strictEqual(api.mode,'overview','Initial graph must stay in overview');
assert.strictEqual(api.visibleNodeIds.length,data.nodes.length,'Initial overview includes every client');
assert.strictEqual(elements.get('gidSearch').value,'','Initial search stays empty');
assert.strictEqual(elements.get('egoButton').disabled,true,'Node view needs an explicit selection');
assert.strictEqual(elements.get('hopSelect').disabled,true,'Hop control needs an explicit selection');
assert(elements.get('topRows').childNodes.every(row=>!row.className.split(/\s+/).includes('selected')),'No initial selected row');
assert.strictEqual(elements.get('cardHeading').focusCalls.length,0,'Initial overview does not steal focus');
assert.strictEqual(elements.get('cardHeading').scrollCalls.length,0,'Initial overview does not scroll');
function assertPriorityPosition(){
 const children=card.childNodes;
 const evidenceIndex=children.findIndex(e=>e.className==='evidence');
 const headings=children.filter(e=>e.className==='subheading'&&e.textContent==='Состав приоритета');
 assert.strictEqual(headings.length,1,'No duplicate priority breakdown');
 const headingIndex=children.indexOf(headings[0]);
 assert.strictEqual(headingIndex,evidenceIndex+1,'Priority heading must immediately follow evidence');
 const generalMetricsIndex=children.findIndex(e=>e.className==='metrics'&&e.childNodes.some(m=>m.childNodes?.[0]?.textContent==='Приоритет проверки'));
 assert(generalMetricsIndex>headingIndex,'Priority explanation must precede general metrics');
}
elements.get('firstNode').click();
assert.strictEqual(api.selected,nodes[0].gid,'First candidate is selected only after a click');
assert.strictEqual(api.mode,'ego','Explicit first-candidate selection opens its connections');
assert.strictEqual(elements.get('gidSearch').value,nodes[0].gid);
assert.strictEqual(elements.get('egoButton').disabled,false);
assert(card.textContent.includes(nodes[0].gid));
assertPriorityPosition();

// These two counterexamples must explain why the higher-priority coordinator rule failed.
// Select by data conditions, not by preselected opaque gid.
const thresholds=data.thresholds;
const distributorCounter=data.nodes.find(n=>n.role==='distributor'&&n.in_deg>0&&n.out_deg>0&&n.seed_reach>=thresholds.coordinator_seed_reach&&n.betweenness>=thresholds.coordinator_betweenness_actual&&n.structural_neighbors<thresholds.coordinator_structural_neighbors);
const peripheralCounter=data.nodes.find(n=>n.role==='peripheral'&&n.in_deg>0&&n.out_deg>0&&n.seed_reach>=thresholds.coordinator_seed_reach&&n.structural_neighbors>=thresholds.coordinator_structural_neighbors&&n.betweenness<thresholds.coordinator_betweenness_actual);
assert(distributorCounter,'Real distributor counterexample with insufficient structural neighbors');
assert(peripheralCounter,'Real peripheral counterexample below the q99 threshold');
const cases=[nodes[0],data.nodes.find(n=>n.cluster_stability_scope==='isolate_not_assessed'),data.nodes.find(n=>n.depth===4),data.nodes.find(n=>n.role==='transit'),data.nodes.find(n=>n.is_seed&&n.out_deg>0),distributorCounter,peripheralCounter,data.nodes.find(n=>n.role==='consolidator'),data.nodes.find(n=>n.role==='terminal')];
const roleOrderText='Порядок: Координатор → Консолидатор → Распределитель → Транзитный → Получатель → Периферийный.';
const nf=new Intl.NumberFormat('ru-RU');
function descendants(e){return[e,...e.childNodes.flatMap(descendants)]}
function metric(label){const m=descendants(card).find(e=>e.className==='metric'&&e.childNodes[0].textContent===label);assert(m,'Missing metric '+label);return m.childNodes[1].textContent}
for(const n of cases){
 assert(api.selectNode(n.gid));assertPriorityPosition();assert.strictEqual(api.selected,n.gid);assert.strictEqual(typeof n.gid,'string');assert(api.visibleNodeIds.includes(n.gid));
 assert.strictEqual(metric('Узлы сбора/рассылки рядом'),nf.format(n.structural_neighbors));
 assert.strictEqual(metric('Betweenness'),Number(n.betweenness).toFixed(6));
 const cycle=n.in_cycle==null?'н/д':n.in_cycle?'да':'нет';
 assert.strictEqual(metric('Участие в направленном цикле'),cycle);
 if(n.role==='consolidator'&&n.depth<4)assert(!card.textContent.includes('На 4-м колене сила правила снижена'),'Ordinary consolidator must not claim truncated data');
 assert(card.textContent.includes(roleOrderText));
 assert(card.textContent.includes(`betweenness ≥ ${Number(thresholds.coordinator_betweenness_actual).toFixed(6)} (q99) и > 0.`));
 assert(card.textContent.includes(`соседей сбора/рассылки ≥ ${thresholds.coordinator_structural_neighbors}`));
 assert(card.textContent.includes(`seed по структуре без дат ≥ ${thresholds.coordinator_seed_reach}`));
 if(n.role!=='coordinator')assert(card.textContent.includes('Проверка координатора:'));
 assert.strictEqual(metric('Seed по структуре · без дат'),nf.format(n.seed_reach));
 assert.strictEqual(metric('Seed · только следующие дни'),nf.format(n.temporal_seed_reach_strict_days));
 assert.strictEqual(metric('Seed · верхняя граница по датам'),nf.format(n.temporal_seed_reach_upper));
 assert.strictEqual(metric('Место при смене весов'),`${nf.format(n.priority_min_rank)}–${nf.format(n.priority_max_rank)}`);
 assert.strictEqual(metric('В топ-20 при смене весов'),(100*n.priority_top20_frequency).toFixed(0)+'%');
 assert(card.textContent.includes(n.next_request));assert(card.textContent.includes('Все сработавшие правила:'));assert(card.textContent.includes('Состав приоритета'));
 const download=descendants(card).find(e=>e.tagName==='button'&&e.textContent==='Скачать карточку ↓');assert(download);download.click();
 assert(downloaded.includes(roleOrderText));assert(downloaded.includes(`Узлы сбора/рассылки рядом: ${n.structural_neighbors}`));assert(downloaded.includes(`Betweenness: ${Number(n.betweenness).toFixed(6)}`));assert(downloaded.includes(`betweenness ≥ ${Number(thresholds.coordinator_betweenness_actual).toFixed(6)} (q99) и > 0.`));
 assert(downloaded.includes(`Участие в направленном цикле: ${cycle}`));
 assert(downloaded.includes(`Сопоставлено за 1–2 дня: ${(100*n.fast_out_share).toFixed(0)}%`));
 assert(downloaded.includes(`Макс. плательщиков за день: ${nf.format(n.max_payers_same_day)}`));
 const limits=descendants(card).find(e=>e.className==='limits');assert(limits);
 for(const line of limits.childNodes)assert(downloaded.includes(line.textContent),'TXT omits visible limitation: '+line.textContent);
 assert(limits.textContent.includes('Видны только внутрибанковские переводы от 5 000 ₸ за июль 2026.'));
 assert(card.textContent.includes('Роль — гипотеза для проверки.'));assert(downloaded.includes('порядок переводов внутри дня неизвестен.'));
 if(n.base_role!==n.role){const base=card.childNodes.find(e=>e.textContent.startsWith('Базовая роль по поведению:'));assert(base);assert(downloaded.includes(base.textContent));}
 assert(downloaded.includes(n.gid));assert(downloaded.includes(n.next_request));assert(downloaded.includes(`Seed по структуре, без учёта дат: ${n.seed_reach}`));assert(downloaded.includes(`Seed по строго возрастающим дням: ${n.temporal_seed_reach_strict_days}`));assert(downloaded.includes('Все сработавшие правила:'));
 if(n.cluster_stability_scope==='isolate_not_assessed'){assert.strictEqual(metric('Сходство состава кластера'),'Не оценивалась');assert(downloaded.includes('Сходство состава кластера: Не оценивалась'));assert(downloaded.includes('Изолят: устойчивость кластера не оценивалась.'));assert.strictEqual(api.visibleNodeIds.length,1)}
 else assert.strictEqual(metric('Сходство состава кластера'),(100*n.cluster_stability).toFixed(0)+'%');
 if(n.is_seed)assert(card.textContent.includes('Сумма вкладов умножена на'));
}
const prior=api.selected;assert.strictEqual(api.selectNode('000000000000000000'),false);assert.strictEqual(api.selected,prior);assert(elements.get('status').textContent.includes('не найден'));
api.selectNode(nodes[0].gid);elements.get('hopSelect').value='2';elements.get('hopSelect').listeners.change();assert(api.visibleNodeIds.length>1);
const beforeFilterCard=card.textContent;
elements.get('roleFilter').value='transit';elements.get('roleFilter').listeners.change();assert(api.visibleNodeIds.every(g=>data.nodes.find(n=>n.gid===g).role==='transit'));
const warning=elements.get('selectedOutsideFilter');
assert.strictEqual(api.selected,nodes[0].gid);assert.strictEqual(card.textContent,beforeFilterCard);assert(warning.textContent.includes('вне текущего фильтра'));assert(!warning.classList.contains('hidden'));
elements.get('egoButton').click();assert.strictEqual(api.mode,'ego');assert(api.visibleNodeIds.includes(api.selected));assert(warning.classList.contains('hidden'));assert.strictEqual(warning.textContent,'');
elements.get('roleFilter').value='';elements.get('roleFilter').listeners.change();assert.strictEqual(api.mode,'overview');assert(warning.classList.contains('hidden'));
const otherCluster=data.nodes.find(n=>String(n.cluster_id)!==String(nodes[0].cluster_id));assert(otherCluster);
elements.get('clusterFilter').value=String(otherCluster.cluster_id);elements.get('clusterFilter').listeners.change();assert(!api.visibleNodeIds.includes(api.selected));assert(!warning.classList.contains('hidden'));assert.strictEqual(card.textContent,beforeFilterCard);
elements.get('clusterFilter').value='';elements.get('clusterFilter').listeners.change();assert(api.visibleNodeIds.includes(api.selected));assert(warning.classList.contains('hidden'));
const heading=elements.get('cardHeading'),focusBefore=heading.focusCalls.length,scrollBefore=heading.scrollCalls.length;
viewportWidth=700;api.selectNode(nodes[1].gid);assert.strictEqual(heading.focusCalls.length,focusBefore+1);assert.strictEqual(heading.focusCalls.at(-1).preventScroll,true);assert.strictEqual(heading.scrollCalls.length,scrollBefore+1);assert.strictEqual(heading.scrollCalls.at(-1).block,'start');
viewportWidth=1200;api.selectNode(nodes[0].gid);assert.strictEqual(heading.focusCalls.length,focusBefore+1);assert.strictEqual(heading.scrollCalls.length,scrollBefore+1);
// Source-level CSS presence only: this does not measure visibility or layout in a browser.
for(const csv of ['nodes_roles.csv','clusters.csv','top_nodes.csv'])assert(html.includes(`href="${csv}" download`));
assert(!/\.header-right(?:\s+\.exports)?\{[^}]*display:\s*none/.test(html),'Narrow viewport CSS must not hide CSV links with their parent');
assert(data.nodes.every(n=>typeof n.gid==='string'));assert(data.edges.every(e=>typeof e.src==='string'&&typeof e.dst==='string'));
console.log('PASS: initial overview without selection, manual first-candidate selection, priority directly after evidence with no duplicates; JavaScript parses and initializes; 9 real node cards/downloads covering all six roles; cycle/shared limits/base role in TXT; filtered-selection warning/reset; narrow focus/scroll calls; distributor/peripheral coordinator counterexamples; all roles expose structural neighbors/betweenness and rule precedence; string gids; static/temporal reach; all matched rules; exact next request; isolate no stability claim; rank sensitivity; priority contributions/seed discount; unknown gid; 2 hops; transit filter.');
function ask(q){elements.get('assistantQuestion').value=q;elements.get('assistantForm').listeners.submit({preventDefault(){}});return elements.get('assistantAnswer').textContent}
function unchanged(q,expected){const before={selected:api.selected,mode:api.mode,card:card.textContent,visible:JSON.stringify(api.visibleNodeIds)};const answer=ask(q);assert(expected.test(answer),`${q}: ${answer}`);assert.strictEqual(api.selected,before.selected,q);assert.strictEqual(api.mode,before.mode,q);assert.strictEqual(card.textContent,before.card,q);assert.strictEqual(JSON.stringify(api.visibleNodeIds),before.visible,q);assert.strictEqual(elements.get('assistantAnswer').childNodes.length,0,'No misleading answer cards: '+q)}
const gid=nodes[0].gid,other=nodes[1].gid,unknown='999999999999999999';assert(!data.nodes.some(n=>n.gid===unknown));
for(const q of ['',`Какой доход у клиента ${gid}?`,`Зачем ${gid} перевёл деньги?`,`Путь от ${gid} до ${other}`,`Общие получатели ${gid} и ${other}`,'Объясни','Объясни gid','Кластер 1','Топ 20','первый','Не показывай топ',`Объясни ${gid},`,`Объясни ${gid} и`,`Объясни ${gid} лишнее`,`${gid}suffix`,gid+'123456','какие ограничения у данных?'])unchanged(q,/Произвольные вопросы не поддерживаются/);
for(const q of [unknown,`Объясни ${unknown}`,`Объясни ${gid} и ${unknown}`,`${gid}, ${unknown}, ${other}`,`топ ${unknown}`,`Какой доход у ${unknown}?`])unchanged(q,/gid отсутствует в выгрузке/);
unchanged(nodes.slice(0,4).map(n=>n.gid).join(' '),/В запросе 4 разных gid.*Лимит — 3/);
for(const q of [gid,`Объясни ${gid}`,`ОБЪЯСНИ ${gid}!`,`${gid}, ${other}`,`Объясни ${gid} и ${other}`,`Объясни ${gid}; ${other}`,`${gid} ${other}`,`${gid} ${gid} ${other}`]){const answer=ask(q);assert(answer.includes(gid));assert.strictEqual(api.selected,gid);assert.strictEqual(api.mode,'ego');const ids=[...new Set(q.match(/[0-9]{15,20}/g))];assert.strictEqual(elements.get('assistantAnswer').childNodes.length,ids.length);ids.forEach(id=>assert(answer.includes(id)));}
for(const q of ['топ','ТОП!','кластеры','Ограничения.']){const before=api.selected,answer=ask(q);assert(!answer.includes('Произвольные вопросы'));assert.strictEqual(api.selected,before);if(/^топ/i.test(q))assert.strictEqual(elements.get('assistantAnswer').childNodes.length,3);else assert(answer.length>100)}
ask(`Объясни ${gid} и ${other}`);elements.get('assistantAnswer').childNodes[1].childNodes[0].click();assert.strictEqual(api.selected,other);
console.log('PASS: browser assistant accepts only full commands; 17 unsupported/incomplete + 6 unknown/mixed + over-limit requests preserve card/selection/mode/graph; 8 gid command variants and 4 summary commands work; reply gid buttons navigate.');
