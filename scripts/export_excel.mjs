// Build an Excel report from the shared Python accounting projection.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const require = createRequire(path.resolve(process.env.LG_EXCEL_RUNTIME || '.cache/excel', 'package.json'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: export_excel.mjs data.json output.xlsx');
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const turns = wb.worksheets.add('Turns');
const tree = wb.worksheets.add('Execution tree');
const ref = wb.worksheets.add('Reference data');
const col = n => { let s=''; for(n++; n; n=Math.floor((n-1)/26)) s=String.fromCharCode(65+(n-1)%26)+s; return s; };
const cell = (s,r,c) => s.getRange(`${col(c)}${r}`);
const value = (s,r,c,v) => { cell(s,r,c).values=[[typeof v==='string' && v.startsWith('=') ? "'"+v : v ?? null]]; };
const formula = (s,r,c,f) => { cell(s,r,c).formulas=[[f]]; };
const numeric = x => x == null ? null : Number(x);
const labels=['Fresh input','Cache read','Cache write','Output (non-reasoning)','Reasoning'];
const categoryIndexes=[[0],[1],[2,3,4],[5],[6]];
const catHeaders=labels.flatMap(x=>[`${x} tokens`,`${x} EUR`,`${x} USD`]);
const headers=['Event / request','Turn','Type','Model · effort','Status','Input context tokens','Request / user prompt','Response / tool result','Reasoning text',...catHeaders,'Total EUR','Total USD','Projected EUR','Projected USD','Span ID','Parent ID','Description','Elapsed ms','Request cache-write tokens','Response cache-write tokens','Context after response tokens'];
const treeHeaders=['Operation','Depth','Type','Model · effort','Status','Input context tokens','Description','Span ID','Parent ID',...catHeaders,'Total EUR','Total USD','Projected EUR','Projected USD','Request','Elapsed ms','Accounting'];
const rateRows = new Map();
const rateEntries=Object.entries(data.prices.models);

function baseStyle(sheet,cols,rows) {
  const range=sheet.getRange(`A1:${col(cols-1)}${rows}`);
  range.format.font={name:'Arial',size:10,color:'#193e42'};
  range.format.verticalAlignment='top';
  range.format.rowHeight=24;
  range.format.columnWidth=18;
  sheet.getRange(`A1:${col(cols-1)}1`).format={fill:'#173b70',font:{name:'Arial',size:10,bold:true,color:'#ffffff'},wrapText:true,rowHeight:44,verticalAlignment:'center',horizontalAlignment:'center'};
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  sheet.showGridLines=false;
  sheet.tabColor='#173b70';
}
baseStyle(ref,9,30);
ref.getRange('A1:I1').values=[['Parameter','Value','Unit','','','','','','']];
const settings=[
  ['Number of executions',100000,'executions'],
  ['USD → EUR',numeric(data.prices.exchange?.rate),'EUR per USD'],
  ['Cache duration','5m',''],
  ['Run ID',data.run.id,''],
  ['Recording',data.run.demo?'Offline simulation':'Provider trace',''],
  ['FX reference date',data.prices.exchange?.date ? new Date(data.prices.exchange.date+'T00:00:00Z'):null,''],
  ['FX fetched at',data.prices.exchange?.fetched_at ? new Date(data.prices.exchange.fetched_at):null,'UTC'],
  ['FX source',data.prices.exchange?.source || 'Unavailable',''],
  ['Rounding','Only projected costs: ROUND(cost × executions, 0)',''],
  ['Tree totals','Parent rows include descendants; do not sum all tree rows.',''],
  ['Cache writes','Simulation cache movements are shown separately from provider-billed writes.',''],
  ['Pricing scope',data.prices.note,''],
];
for(let i=0;i<settings.length;i++) ref.getRange(`A${i+2}:C${i+2}`).values=[settings[i]];
ref.getRange('B2').format={fill:'#fff1b8',font:{name:'Arial',size:10,bold:true,color:'#173b70'}};
ref.getRange('B2').setNumberFormat('#,##0');
ref.getRange('B3').setNumberFormat('0.###############;-0.###############;0');
ref.getRange('B7').setNumberFormat('yyyy-mm-dd');
ref.getRange('B8').setNumberFormat('yyyy-mm-dd hh:mm:ss');
ref.getRange('B2').dataValidation={rule:{type:'whole',operator:'greaterThanOrEqual',formula1:0}};
ref.getRange('A15:I15').values=[['Model','Input USD / 1M tokens','Cache read USD / 1M tokens','Cache write USD / 1M tokens','Output USD / 1M tokens','Reasoning USD / 1M tokens','Verified date','Source','Notes']];
ref.getRange('A15:I15').format={fill:'#173b70',font:{name:'Arial',size:10,bold:true,color:'#ffffff'},rowHeight:44,wrapText:true};
rateEntries.forEach(([key,rate],i)=>{
  const r=16+i;rateRows.set(key,r);
  ref.getRange(`A${r}:I${r}`).values=[[key,numeric(rate.input),numeric(rate.cache_read),numeric(rate.cache_write ?? rate.cache_write_5m),numeric(rate.output),numeric(rate.output),new Date((rate.as_of || data.prices.as_of)+'T00:00:00Z'),rate.source || data.prices.sources.join('; '),'Cache write assumes 5m when unspecified']];
});
ref.getRange(`B16:F${15+rateEntries.length}`).setNumberFormat('0.###############;-0.###############;0');
ref.getRange(`G16:G${15+rateEntries.length}`).setNumberFormat('yyyy-mm-dd');
ref.getRange(`G16:G${15+rateEntries.length}`).conditionalFormats.addCustom('G16<TODAY()',{fill:'#ffd3d3',font:{color:'#8b1010'}});
ref.getRange('B7').conditionalFormats.addCustom('B7<TODAY()',{fill:'#ffd3d3',font:{color:'#8b1010'}});
ref.getRange('A1:A30').format.columnWidth=34;
ref.getRange('B1:B30').format.columnWidth=52;
ref.getRange('H1:H30').format.columnWidth=65;
ref.getRange('I1:I30').format.columnWidth=40;
ref.getRange('B5:B13').format.wrapText=true;
ref.getRange('B5:B13').format.rowHeight=28;
ref.getRange('B9:B13').format.rowHeight=56;
ref.getRange(`A16:I${15+rateEntries.length}`).format.wrapText=true;
ref.getRange(`A16:I${15+rateEntries.length}`).format.rowHeight=72;

const eventRows=new Map();
const categoryRows=new Map();
const turnRows=new Map();
const sequence=[];
const contents = messages => messages?.map(m=>[
  typeof m.content==='string'?m.content:JSON.stringify(m.content),
  ...(m.tool_calls||[]).map(c=>JSON.stringify(c))
].filter(Boolean).join('\n')).join('\n') || '';
let currentEvent=null;
function line(stage,{tokens=null,text='',total=null,kind='detail',cat=null}={}) {
  const r=sequence.length+2;
  sequence.push({r,stage,tokens,text,total,kind,cat,event:currentEvent});
  return r;
}
line('Run total',{kind:'total'});
for(const e of data.events){
  currentEvent=e;
  const s=e.step;
  if(!turnRows.has(e.turn)) turnRows.set(e.turn,line(`Turn ${e.turn}`,{kind:'turn'}));
  if(s.kind==='tool'){
    line(`Tool · ${s.name}`,{kind:'tool'});
    line('    Arguments',{text:contents(s.request)});
    line('    Result',{text:contents(s.response)});
    continue;
  }
  const rows=new Map();categoryRows.set(s.id,rows);
  const ledger=JSON.parse(s.context?.context_ledger||'{}');
  line('LLM request',{tokens:s.usage?.input_tokens,kind:'request'});
  if(s.usage?.cache_read || e.growth?.previous_tokens!=null)
    rows.set(1,line('Conversation history · cache read',{cat:1}));
  rows.set(0,line('Fresh input',{cat:0}));
  if(e.growth?.previous_tokens==null){
    line('    Tool definitions',{tokens:s.context?.simulated_definitions_tokens});
    const system=(s.request||[]).filter(m=>m.role==='system');
    line('    System prompt',{tokens:s.context?.simulated_system_tokens,text:contents(system)||'Empty system message (framing only).'});
  }
  if(e.user_prompt) line('    User prompt',{tokens:e.user_tokens,text:e.user_prompt});
  const newTools=(s.request||[]).slice(e.growth?.retained||0).filter(m=>m.role==='tool');
  if(newTools.length) line('    Tool result',{tokens:e.input_parts.find(p=>p.label==='Tool-result input')?.tokens,text:contents(newTools)});
  const billedWrite=categoryIndexes[2].some(i=>e.cells[i].tokens||e.cells[i].partial);
  const requestWrite=line('Cache write',{tokens:ledger.request_cache_write_tokens,cat:billedWrite?2:null});
  if(billedWrite)rows.set(2,requestWrite);
  line('Context · added / total',{tokens:ledger.request_cache_write_tokens,total:ledger.request_tokens});
  line('LLM response',{tokens:s.usage?.output_tokens,kind:'response'});
  if(e.cells[6].tokens||e.cells[6].partial||e.thinking){
    rows.set(4,line('Reasoning',{cat:4}));
    if(e.thinking)line('    Reasoning text',{text:e.thinking,kind:'thinking'});
  }
  rows.set(3,line('Output',{cat:3}));
  line('    Output content',{text:contents(s.response)});
  line('Cache write',{tokens:ledger.response_cache_write_tokens});
  line('Context · added / total',{tokens:ledger.response_cache_write_tokens,total:ledger.context_after_response_tokens});
  eventRows.set(s.id,line('Call total',{kind:'calltotal'}));
}
for(const e of data.events.filter(e=>e.step.kind==='model')){
  const stages=sequence.filter(row=>row.event===e&&row.kind!=='turn').map(row=>row.stage);
  for(const [first,second] of [['LLM request','Fresh input'],['Fresh input','LLM response'],['LLM response','Output'],['Output','Call total']]){
    if(stages.indexOf(first)>=stages.indexOf(second))throw new Error(`Incorrect sequence for ${e.label}`);
  }
}
const last=sequence.length+1;
const sequenceHeaders=['Request','Turn','Sequence','Tokens','EUR / execution','USD / execution','Projected EUR','Projected USD','Content','Model · effort','Status','Context total tokens','Span ID'];
baseStyle(turns,sequenceHeaders.length,last);
turns.getRange('A1:M1').values=[sequenceHeaders];
function projects(sheet,r,eurCol,usdCol,outEUR,outUSD){
  formula(sheet,r,outEUR,`=IF(ISNUMBER(${col(eurCol)}${r}),ROUND(${col(eurCol)}${r}*'Reference data'!$B$2,0),"Unknown")`);
  formula(sheet,r,outUSD,`=IF(ISNUMBER(${col(usdCol)}${r}),ROUND(${col(usdCol)}${r}*'Reference data'!$B$2,0),"Unknown")`);
}
function sumRefs(sheet,r,c,refs){
  if(!refs.length){value(sheet,r,c,0);return;}
  formula(sheet,r,c,`=IF(COUNT(${refs.join(',')})=${refs.length},SUM(${refs.join(',')}),"Unknown")`);
}
for(const row of sequence){
  const {r,event:e}=row,s=e?.step;
  turns.getRange(`A${r}:M${r}`).values=[[
    s?.kind==='model'&&row.kind!=='turn'?e.label:null,e?.turn??null,row.stage,row.tokens??null,null,null,null,null,row.text||null,
    row.kind==='request'&&s?.model?`${s.model}${s.effort?'-'+s.effort:''}`:null,
    ['calltotal','tool'].includes(row.kind)?s?.status:null,row.total??null,
    ['request','tool'].includes(row.kind)?s?.id:null,
  ]];
  if(row.cat!=null){
    const indices=categoryIndexes[row.cat];
    value(turns,r,3,s.usage?indices.reduce((n,i)=>n+e.cells[i].tokens,0):'Unknown');
    const known=indices.every(i=>!e.cells[i].partial);
    const key=data.prices.aliases[`${s.provider}:${s.model}`]||`${s.provider}:${s.model}`;
    const rr=rateRows.get(key);
    if(!known||!rr){value(turns,r,4,'Unknown');value(turns,r,5,'Unknown');}
    else {
      if(row.cat===2&&(s.usage?.cache_write_1h||s.usage?.cache_write_5m))
        value(turns,r,5,indices.reduce((n,i)=>n+Number(e.cells[i].usd),0));
      else formula(turns,r,5,`=IF(D${r}=0,0,IF(ISNUMBER('Reference data'!${col(row.cat+1)}${rr}),D${r}*'Reference data'!${col(row.cat+1)}${rr}/1000000,"Unknown"))`);
      formula(turns,r,4,`=IF(AND(ISNUMBER(F${r}),ISNUMBER('Reference data'!$B$3)),F${r}*'Reference data'!$B$3,"Unknown")`);
    }
    projects(turns,r,4,5,6,7);
  }
  if(row.kind==='calltotal'){
    const sources=[...categoryRows.get(s.id).values()];
    for(const c of [4,5])sumRefs(turns,r,c,sources.map(n=>`${col(c)}${n}`));
    projects(turns,r,4,5,6,7);
  }
  if(['total','turn','request','response','calltotal'].includes(row.kind)){
    turns.getRange(`A${r}:M${r}`).format={fill:['total','turn'].includes(row.kind)?'#dbe5f1':'#f0f5fc',font:{bold:true,color:'#173b70'}};
  }else if(row.kind==='tool')turns.getRange(`A${r}:M${r}`).format={fill:'#f4f4f4',font:{bold:true}};
  if(row.cat!=null)cell(turns,r,2).format.font={bold:true};
  if(row.kind==='thinking')cell(turns,r,8).format.font={italic:true};
  if(row.text){
    const lines=String(row.text).split('\n').reduce((n,l)=>n+Math.max(1,Math.ceil(l.length/88)),0);
    turns.getRange(`A${r}:M${r}`).format.rowHeight=Math.max(24,lines*13+8);
  }
}
for(const c of [4,5])sumRefs(turns,2,c,[...eventRows.values()].map(n=>`${col(c)}${n}`));
projects(turns,2,4,5,6,7);
for(const [turn,r]of turnRows){
  for(const c of [4,5])sumRefs(turns,r,c,data.events.filter(e=>e.turn===turn&&e.step.kind==='model').map(e=>`${col(c)}${eventRows.get(e.step.id)}`));
  projects(turns,r,4,5,6,7);
}
turns.getRange(`A1:B${last}`).format.columnWidth=10;
turns.getRange(`C1:C${last}`).format.columnWidth=38;
turns.getRange(`D1:D${last}`).format.columnWidth=14;
turns.getRange(`E1:F${last}`).format.columnWidth=22;
turns.getRange(`G1:H${last}`).format.columnWidth=18;
turns.getRange(`I1:I${last}`).format.columnWidth=100;
turns.getRange(`I2:I${last}`).format.wrapText=true;
turns.getRange(`J1:J${last}`).format.columnWidth=26;
turns.getRange(`M1:M${last}`).format.columnWidth=38;
turns.getRange(`E2:F${last}`).setNumberFormat('0.###############;-0.###############;0');
turns.getRange(`G2:H${last}`).setNumberFormat('#,##0');
turns.getRange(`D2:D${last}`).setNumberFormat('#,##0');
turns.getRange(`L2:L${last}`).setNumberFormat('#,##0');
turns.getRange(`A2:M${last}`).conditionalFormats.addCustom('$K2="error"',{fill:'#ffd3d3',font:{color:'#8b1010'}});

// Tree layout is unchanged; its formulas now point to the sequential category rows.
function rollup(sheet,r,sourceRows){
  const ids=[...eventRows].filter(([,n])=>sourceRows.includes(n)).map(([id])=>id);
  categoryIndexes.forEach((_,i)=>{
    const sources=ids.map(id=>categoryRows.get(id)?.get(i)).filter(Boolean);
    for(let j=0;j<3;j++)sumRefs(sheet,r,9+3*i+j,sources.map(n=>`'Turns'!${col(3+j)}${n}`));
  });
  for(let j=0;j<2;j++)sumRefs(sheet,r,24+j,sourceRows.map(n=>`'Turns'!${col(4+j)}${n}`));
  projects(sheet,r,24,25,26,27);
}
function moneyFormat(sheet,rows){
  for(const c of ['K','L','N','O','Q','R','T','U','W','X','Y','Z'])sheet.getRange(`${c}2:${c}${rows}`).setNumberFormat('0.###############;-0.###############;0');
  sheet.getRange(`AA2:AB${rows}`).setNumberFormat('#,##0');
  for(const c of ['F','J','M','P','S','V'])sheet.getRange(`${c}2:${c}${rows}`).setNumberFormat('#,##0');
}

const treeLast=data.tree.length+1;
baseStyle(tree,treeHeaders.length,treeLast);
tree.getRange(`A1:${col(treeHeaders.length-1)}1`).values=[treeHeaders];
const byId=new Map(data.tree.map(t=>[t.step.id,t]));
for(let i=0;i<data.tree.length;i++){
  const t=data.tree[i],s=t.step,r=i+2;
  const event=data.events.find(e=>e.step.id===s.id);
  tree.getRange(`A${r}:I${r}`).values=[['    '.repeat(t.depth)+s.name,t.depth,s.kind,s.model?`${s.model}${s.effort?'-'+s.effort:''}`:'',s.status,s.usage?.input_tokens??null,t.description,s.id,s.parent_id]];
  const descendants=data.events.filter(e=>{
    if(e.step.kind!=='model') return false;
    let id=e.step.id;
    while(id){if(id===s.id)return true;id=byId.get(id)?.step.parent_id;}
    return false;
  }).map(e=>eventRows.get(e.step.id));
  if(descendants.length) rollup(tree,r,descendants,'Turns');
  value(tree,r,28,event?.step.kind==='model'?event.label:null);
  value(tree,r,29,(s.end_ns-s.start_ns)/1e6);
  value(tree,r,30,s.kind==='model'?'Own call':descendants.length?'Subtree total':'No model charge');
  tree.getRange(`A${r}:I${r}`).format.fill=s.kind==='model'?'#f0f5fc':t.depth===0?'#e8eef7':'#ffffff';
  tree.getRange(`A${r}:${col(treeHeaders.length-1)}${r}`).format.rowHeight=54;
}
moneyFormat(tree,treeLast);
tree.getRange(`A1:A${treeLast}`).format.columnWidth=48;
tree.getRange(`B1:C${treeLast}`).format.columnWidth=12;
tree.getRange(`D1:D${treeLast}`).format.columnWidth=26;
tree.getRange(`G1:G${treeLast}`).format.columnWidth=62;
tree.getRange(`A2:A${treeLast}`).format.wrapText=true;
tree.getRange(`G2:G${treeLast}`).format.wrapText=true;
tree.getRange(`H1:I${treeLast}`).format.columnWidth=38;
tree.getRange(`A2:${col(treeHeaders.length-1)}${treeLast}`).conditionalFormats.addCustom('$E2="error"',{fill:'#ffd3d3',font:{color:'#8b1010'}});
tree.getRange(`AD2:AD${treeLast}`).setNumberFormat('0.00');


wb.recalculate();
const expected=Number(data.expected_usd);
const actual=cell(turns,2,5).values[0][0];
if(Math.abs(actual-expected)>1e-10) throw new Error(`USD mismatch ${actual} vs ${expected}`);
const initial=cell(turns,2,6).values[0][0];
value(ref,2,1,200000);wb.recalculate();
const changed=cell(turns,2,6).values[0][0];
const expectedChanged=Math.round(expected*Number(data.prices.exchange.rate)*200000);
if(Math.abs(changed-expectedChanged)>0.00001) throw new Error('Execution multiplier does not recalculate');
value(ref,2,1,100000);wb.recalculate();
console.log(JSON.stringify({runUSD:actual,projectedEUR:initial,projectionAt200000:changed,rows:last,treeRows:treeLast}));
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:10},maxChars:1500})).ndjson);
console.log((await wb.inspect({kind:'table',range:'Turns!E1:H5',include:'values,formulas',tableMaxRows:5,tableMaxCols:4,maxChars:2200})).ndjson);
await fs.mkdir(path.dirname(output),{recursive:true});
for(const [name,range,file] of [['Turns','A1:H22','turns'],['Turns','C7:I22','content'],['Execution tree','A1:G9','tree'],['Reference data','A1:D8','reference']]){
  const image=await wb.render({sheetName:name,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(path.dirname(output),`${file}-preview.png`),new Uint8Array(await image.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(output);
console.log(output);
