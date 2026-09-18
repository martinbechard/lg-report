// Copyright (c) 2026 Martin.Bechard@DevConsult.ca
// Build an editable Excel estimate from the shared Python accounting projection.
// Keep one-call costs unrounded; only the execution forecast rounds to whole
// currency units. Python owns trace interpretation and the original tariff data.
// AI attribution: Generated with AI assistance.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const require = createRequire(path.resolve(process.env.LG_EXCEL_RUNTIME || '.cache/excel', 'package.json'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const [input, output] = process.argv.slice(2);
// Both paths are required: do not create a workbook from implicit input or write
// to an accidental destination when either CLI argument is absent.
if (!input || !output) throw new Error('Usage: export_excel.mjs data.json output.xlsx');
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const turns = wb.worksheets.add('Turns');
const tree = wb.worksheets.add('Execution tree');
const ref = wb.worksheets.add('Reference data');
/** Convert a zero-based column index to Excel letters (0 → A, 26 → AA). */
const columnLetter = columnIndex => {
  let letters = '';
  // Excel addresses use digits 1..26, with no zero digit. Subtracting one
  // before each remainder keeps the Z → AA boundary correct.
  for (let ordinal = columnIndex + 1; ordinal; ordinal = Math.floor((ordinal - 1) / 26)) {
    letters = String.fromCharCode(65 + (ordinal - 1) % 26) + letters;
  }
  return letters;
};
/** Resolve a one-based row and zero-based column to a writable cell range. */
const workbookCell = (sheet, rowNumber, columnIndex) => sheet.getRange(`${columnLetter(columnIndex)}${rowNumber}`);
// Captured prompts and tool output are untrusted cell data. Only writeFormula() may
// create executable spreadsheet formulas; a leading '=' in content stays literal.
const writeCellValue = (sheet, rowNumber, columnIndex, content) => {
  const literal = typeof content === 'string' && content.startsWith('=') ? "'" + content : content ?? null;
  workbookCell(sheet, rowNumber, columnIndex).values = [[literal]];
};
/** Write a formula produced by this exporter, never captured message content. */
const writeFormula = (sheet, rowNumber, columnIndex, expression) => {
  workbookCell(sheet, rowNumber, columnIndex).formulas = [[expression]];
};
// Null denotes unavailable telemetry/rates; Number(null) would falsely make it zero.
const optionalNumber = sourceValue => sourceValue == null ? null : Number(sourceValue);
const labels=['Fresh input','Cache read','Cache write','Output (non-reasoning)','Reasoning'];
// Collapse cache lifetimes only for presentation. Their recorded charges still
// retain the applicable tariff when a call contains explicit lifetime buckets.
const categoryIndexes=[[0],[1],[2,3,4],[5],[6]];
const catHeaders=labels.flatMap(x=>[`${x} tokens`,`${x} EUR`,`${x} USD`]);
const treeHeaders=['Operation','Depth','Type','Model · effort','Status','Input context tokens','Description','Span ID','Parent ID',...catHeaders,'Total EUR','Total USD','Projected EUR','Projected USD','Request','Elapsed ms','Accounting'];
const rateRows = new Map();
const rateEntries=Object.entries(data.prices.models);

// Freeze both identifiers and headers: wide token/currency columns must remain
// understandable while the reader scrolls through a long conversation.
function styleWorksheet(sheet,cols,rows) {
  const range=sheet.getRange(`A1:${columnLetter(cols-1)}${rows}`);
  range.format.font={name:'Arial',size:10,color:'#193e42'};
  range.format.verticalAlignment='top';
  range.format.rowHeight=24;
  range.format.columnWidth=18;
  sheet.getRange(`A1:${columnLetter(cols-1)}1`).format={fill:'#173b70',font:{name:'Arial',size:10,bold:true,color:'#ffffff'},wrapText:true,rowHeight:44,verticalAlignment:'center',horizontalAlignment:'center'};
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  sheet.showGridLines=false;
  sheet.tabColor='#173b70';
}
styleWorksheet(ref,9,30);
ref.getRange('A1:I1').values=[['Parameter','Value','Unit','','','','','','']];
// Recording mode labels simulated data explicitly. Date objects are created only
// when provenance exists; missing timestamps stay blank rather than becoming epoch.
const settings=[
  ['Number of executions',100000,'executions'],
  ['USD → EUR',optionalNumber(data.prices.exchange?.rate),'EUR per USD'],
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
  ref.getRange(`A${r}:I${r}`).values=[[key,optionalNumber(rate.input),optionalNumber(rate.cache_read),optionalNumber(rate.cache_write ?? rate.cache_write_5m),optionalNumber(rate.output),optionalNumber(rate.output),new Date((rate.as_of || data.prices.as_of)+'T00:00:00Z'),rate.source || data.prices.sources.join('; '),'Cache write assumes 5m when unspecified']];
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

// These maps connect sheet formulas to the rows that own charges. A span ID
// maps to its call-total row and its category rows; a turn number maps to its
// heading/total row. Context-detail rows deliberately never enter these maps.
const eventRows=new Map();
const categoryRows=new Map();
const turnRows=new Map();
// Each row follows the conversation's request → context → response sequence.
// Totals reference charge rows, never the repeated explanatory context counts.
const sequence=[];
// Preserve plain message text; serialize structured content and tool-call objects
// so both provider response shapes remain readable. Empty parts add no blank blocks.
const messageText = messages => messages?.map(m=>[
  typeof m.content==='string'?m.content:JSON.stringify(m.content),
  ...(m.tool_calls||[]).map(c=>JSON.stringify(c))
].filter(Boolean).join('\n')).join('\n') || '';
let currentEvent=null;
/** Append one ordered presentation row for the current event.
 * cat is a display category index (0..4), or null for noncharge context/detail.
 * tokens and total are counts, not prices; money formulas are attached later.
 */
function appendSequenceRow(stage,{tokens=null,text='',total=null,kind='detail',cat=null}={}) {
  const r=sequence.length+2;
  sequence.push({r,stage,tokens,text,total,kind,cat,event:currentEvent});
  return r;
}
appendSequenceRow('Run total',{kind:'total'});
for(const e of data.events){
  currentEvent=e;
  const s=e.step;
  // Insert each turn heading once, even when several model/tool events share it.
  if(!turnRows.has(e.turn)) turnRows.set(e.turn,appendSequenceRow(`Turn ${e.turn}`,{kind:'turn'}));
  // Tool execution is a nonbillable event: show arguments/results, then bypass
  // all model charge rows so tool activity cannot inflate the estimate.
  if(s.kind==='tool'){
    appendSequenceRow(`Tool · ${s.name}`,{kind:'tool'});
    appendSequenceRow('    Arguments',{text:messageText(s.request)});
    appendSequenceRow('    Result',{text:messageText(s.response)});
    continue;
  }
  const rows=new Map();categoryRows.set(s.id,rows);
  const ledger=JSON.parse(s.context?.context_ledger||'{}');
  appendSequenceRow('LLM request',{tokens:s.usage?.input_tokens,kind:'request'});
  // Show history when actual cache reads exist OR a prior request establishes
  // history; the latter preserves a meaningful zero-cache-read observation.
  if(s.usage?.cache_read || e.growth?.previous_tokens!=null)
    rows.set(1,appendSequenceRow('Conversation history · cache read',{cat:1}));
  rows.set(0,appendSequenceRow('Fresh input',{cat:0}));
  // No previous context means setup enters here; later calls already carry
  // definitions/system messages in history and must not display them as new.
  if(e.growth?.previous_tokens==null){
    appendSequenceRow('    Tool definitions',{tokens:s.context?.simulated_definitions_tokens});
    const system=(s.request||[]).filter(m=>m.role==='system');
    appendSequenceRow('    System prompt',{tokens:s.context?.simulated_system_tokens,text:messageText(system)||'Empty system message (framing only).'});
  }
  // The Python projection attaches the user prompt only where it starts a turn.
  if(e.user_prompt) appendSequenceRow('    User prompt',{tokens:e.user_tokens,text:e.user_prompt});
  // Slice off retained history before selecting tool messages; only newly
  // appended results belong under this request's fresh input.
  const newTools=(s.request||[]).slice(e.growth?.retained||0).filter(m=>m.role==='tool');
  // Avoid a tool-result row when no new result was submitted to this request.
  if(newTools.length) appendSequenceRow('    Tool result',{tokens:e.input_parts.find(p=>p.label==='Tool-result input')?.tokens,text:messageText(newTools)});
  // A positive count or incomplete category needs a charge row; simulated
  // context movement alone is not evidence of a provider cache-write charge.
  const billedWrite=categoryIndexes[2].some(i=>e.cells[i].tokens||e.cells[i].partial);
  const requestWrite=appendSequenceRow('Cache write',{tokens:ledger.request_cache_write_tokens,cat:billedWrite?2:null});
  if(billedWrite)rows.set(2,requestWrite);
  appendSequenceRow('Context · added / total',{tokens:ledger.request_cache_write_tokens,total:ledger.request_tokens});
  appendSequenceRow('LLM response',{tokens:s.usage?.output_tokens,kind:'response'});
  // Reasoning appears when measured, unpriced/unknown, or text is captured;
  // zero known tokens with no text would contribute only visual clutter.
  if(e.cells[6].tokens||e.cells[6].partial||e.thinking){
    rows.set(4,appendSequenceRow('Reasoning',{cat:4}));
    // Counts cannot supply reasoning text: omit the text row unless captured.
    if(e.thinking)appendSequenceRow('    Reasoning text',{text:e.thinking,kind:'thinking'});
  }
  rows.set(3,appendSequenceRow('Output',{cat:3}));
  appendSequenceRow('    Output content',{text:messageText(s.response)});
  appendSequenceRow('Cache write',{tokens:ledger.response_cache_write_tokens});
  appendSequenceRow('Context · added / total',{tokens:ledger.response_cache_write_tokens,total:ledger.context_after_response_tokens});
  eventRows.set(s.id,appendSequenceRow('Call total',{kind:'calltotal'}));
}
for(const e of data.events.filter(e=>e.step.kind==='model')){
  const stages=sequence.filter(row=>row.event===e&&row.kind!=='turn').map(row=>row.stage);
  for(const [first,second] of [['LLM request','Fresh input'],['Fresh input','LLM response'],['LLM response','Output'],['Output','Call total']]){
    // Reject an inverted required sequence before export; spreadsheet row order
    // must follow the same request-to-response story as the HTML report.
    if(stages.indexOf(first)>=stages.indexOf(second))throw new Error(`Incorrect sequence for ${e.label}`);
  }
}
const last=sequence.length+1;
const sequenceHeaders=['Request','Turn','Sequence','Tokens','EUR / execution','USD / execution','Projected EUR','Projected USD','Content','Model · effort','Status','Context total tokens','Span ID'];
styleWorksheet(turns,sequenceHeaders.length,last);
turns.getRange('A1:M1').values=[sequenceHeaders];
// Scaling before rounding is essential: sub-cent call costs would otherwise
// become zero before a 100,000-execution forecast. Keep source columns precise.
function writeProjectedCosts(sheet, rowNumber, eurColumn, usdColumn, projectedEurColumn, projectedUsdColumn) {
  writeFormula(sheet,rowNumber,projectedEurColumn,`=IF(ISNUMBER(${columnLetter(eurColumn)}${rowNumber}),ROUND(${columnLetter(eurColumn)}${rowNumber}*'Reference data'!$B$2,0),"Unknown")`);
  writeFormula(sheet,rowNumber,projectedUsdColumn,`=IF(ISNUMBER(${columnLetter(usdColumn)}${rowNumber}),ROUND(${columnLetter(usdColumn)}${rowNumber}*'Reference data'!$B$2,0),"Unknown")`);
}
// SUM alone ignores text cells, which would turn an unknown charge into a
// deceptively complete total. Require every referenced charge to be numeric.
function sumKnownCharges(sheet, rowNumber, columnIndex, sourceReferences) {
  // No contributing calls means a genuine zero subtotal, not unknown usage.
  if (!sourceReferences.length) {
    writeCellValue(sheet, rowNumber, columnIndex, 0);
    return;
  }
  const references = sourceReferences.join(',');
  writeFormula(sheet,rowNumber,columnIndex,`=IF(COUNT(${references})=${sourceReferences.length},SUM(${references}),"Unknown")`);
}
for(const row of sequence){
  const {r,event:e}=row,s=e?.step;
  // Request labels belong to model detail rows, model/effort to request headers,
  // status to completed calls/tools, and span IDs to request/tool headers.
  // Leave other cells blank to avoid repeating metadata on every context row.
  turns.getRange(`A${r}:M${r}`).values=[[
    s?.kind==='model'&&row.kind!=='turn'?e.label:null,e?.turn??null,row.stage,row.tokens??null,null,null,null,null,row.text||null,
    row.kind==='request'&&s?.model?`${s.model}${s.effort?'-'+s.effort:''}`:null,
    ['calltotal','tool'].includes(row.kind)?s?.status:null,row.total??null,
    ['request','tool'].includes(row.kind)?s?.id:null,
  ]];
  // Only category rows carry charges; explanatory token/context rows must not
  // acquire formulas that would bill the same tokens a second time.
  if(row.cat!=null){
    const indices=categoryIndexes[row.cat];
    // Missing usage must remain Unknown; summing empty buckets would imply zero.
    writeCellValue(turns,r,3,s.usage?indices.reduce((n,i)=>n+e.cells[i].tokens,0):'Unknown');
    const known=indices.every(i=>!e.cells[i].partial);
    const key=data.prices.aliases[`${s.provider}:${s.model}`]||`${s.provider}:${s.model}`;
    const tariffRow=rateRows.get(key);
    // Either incomplete category data or a missing tariff prevents reliable
    // pricing; preserve Unknown in both currencies instead of a partial total.
    if(!known||!tariffRow){writeCellValue(turns,r,4,'Unknown');writeCellValue(turns,r,5,'Unknown');}
    else {
      // The single editable cache rate cannot represent mixed lifetimes; keep
      // Python's exact category charge for calls with explicit lifetime usage.
      if(row.cat===2&&(s.usage?.cache_write_1h||s.usage?.cache_write_5m))
        writeCellValue(turns,r,5,indices.reduce((n,i)=>n+Number(e.cells[i].usd),0));
      else writeFormula(turns,r,5,`=IF(D${r}=0,0,IF(ISNUMBER('Reference data'!${columnLetter(row.cat+1)}${tariffRow}),D${r}*'Reference data'!${columnLetter(row.cat+1)}${tariffRow}/1000000,"Unknown"))`);
      writeFormula(turns,r,4,`=IF(AND(ISNUMBER(F${r}),ISNUMBER('Reference data'!$B$3)),F${r}*'Reference data'!$B$3,"Unknown")`);
    }
    writeProjectedCosts(turns,r,4,5,6,7);
  }
  // Call totals sum category charge rows only, excluding repeated context sizes.
  if(row.kind==='calltotal'){
    const sources=[...categoryRows.get(s.id).values()];
    for(const c of [4,5])sumKnownCharges(turns,r,c,sources.map(n=>`${columnLetter(c)}${n}`));
    writeProjectedCosts(turns,r,4,5,6,7);
  }
  // Highlight sequence boundaries; totals/turns get stronger fill than model
  // boundaries, while tool rows stay neutral to distinguish them from LLM calls.
  if(['total','turn','request','response','calltotal'].includes(row.kind)){
    turns.getRange(`A${r}:M${r}`).format={fill:['total','turn'].includes(row.kind)?'#dbe5f1':'#f0f5fc',font:{bold:true,color:'#173b70'}};
  }else if(row.kind==='tool')turns.getRange(`A${r}:M${r}`).format={fill:'#f4f4f4',font:{bold:true}};
  // Bold charge labels so they remain visually distinct from subordinate
  // context details; cat=0 (fresh input) is valid and must not be treated as false.
  if(row.cat!=null)workbookCell(turns,r,2).format.font={bold:true};
  // Italicize only exposed reasoning, leaving actual answer/tool content plain.
  if(row.kind==='thinking')workbookCell(turns,r,8).format.font={italic:true};
  // Content rows need height based on wrapped lines; blank metadata rows keep
  // the compact default instead of reserving empty text space.
  if(row.text){
    const lines=String(row.text).split('\n').reduce((n,l)=>n+Math.max(1,Math.ceil(l.length/88)),0);
    turns.getRange(`A${r}:M${r}`).format.rowHeight=Math.max(24,lines*13+8);
  }
}
for(const c of [4,5])sumKnownCharges(turns,2,c,[...eventRows.values()].map(n=>`${columnLetter(c)}${n}`));
writeProjectedCosts(turns,2,4,5,6,7);
// Select only model calls in this turn; tool events have no call-total charge.
for(const [turn,r]of turnRows){
  for(const c of [4,5])sumKnownCharges(turns,r,c,data.events.filter(e=>e.turn===turn&&e.step.kind==='model').map(e=>`${columnLetter(c)}${eventRows.get(e.step.id)}`));
  writeProjectedCosts(turns,r,4,5,6,7);
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

// Reference the same charge rows as Turns so both views recalculate together.
// Subtree rows repeat descendant costs for drill-down; they are not new charges
// and must never be added to the run total.
function writeSubtreeCosts(sheet,r,sourceRows){
  // Keep only model charge rows selected by this subtree. Missing category row
  // references are omitted because optional zero categories have no sheet row.
  const ids=[...eventRows].filter(([,n])=>sourceRows.includes(n)).map(([id])=>id);
  categoryIndexes.forEach((_,i)=>{
    const sources=ids.map(id=>categoryRows.get(id)?.get(i)).filter(Boolean);
    for(let j=0;j<3;j++)sumKnownCharges(sheet,r,9+3*i+j,sources.map(n=>`'Turns'!${columnLetter(3+j)}${n}`));
  });
  for(let j=0;j<2;j++)sumKnownCharges(sheet,r,24+j,sourceRows.map(n=>`'Turns'!${columnLetter(4+j)}${n}`));
  writeProjectedCosts(sheet,r,24,25,26,27);
}
function formatTreeAmounts(sheet,rows){
  for(const c of ['K','L','N','O','Q','R','T','U','W','X','Y','Z'])sheet.getRange(`${c}2:${c}${rows}`).setNumberFormat('0.###############;-0.###############;0');
  sheet.getRange(`AA2:AB${rows}`).setNumberFormat('#,##0');
  for(const c of ['F','J','M','P','S','V'])sheet.getRange(`${c}2:${c}${rows}`).setNumberFormat('#,##0');
}

const treeLast=data.tree.length+1;
styleWorksheet(tree,treeHeaders.length,treeLast);
tree.getRange(`A1:${columnLetter(treeHeaders.length-1)}1`).values=[treeHeaders];
const byId=new Map(data.tree.map(t=>[t.step.id,t]));
for(let i=0;i<data.tree.length;i++){
  const t=data.tree[i],s=t.step,r=i+2;
  const event=data.events.find(e=>e.step.id===s.id);
  // Non-model spans have no model name; append effort only when recorded,
  // rather than inventing a default effort for provider traces.
  tree.getRange(`A${r}:I${r}`).values=[['    '.repeat(t.depth)+s.name,t.depth,s.kind,s.model?`${s.model}${s.effort?'-'+s.effort:''}`:'',s.status,s.usage?.input_tokens??null,t.description,s.id,s.parent_id]];
  const descendants=data.events.filter(e=>{
    // Only model descendants own costs. Walk parent IDs including self so a
    // model row gets its own charge and containers get only contained charges.
    if(e.step.kind!=='model') return false;
    let id=e.step.id;
    while(id){if(id===s.id)return true;id=byId.get(id)?.step.parent_id;}
    return false;
  }).map(e=>eventRows.get(e.step.id));
  // Empty containers/tools have no model charge, so leave their money cells blank.
  if(descendants.length) writeSubtreeCosts(tree,r,descendants,'Turns');
  // R-labels identify models only. Accounting distinguishes own model charge,
  // container subtotal, and no charge; fill emphasizes models and root boundaries.
  writeCellValue(tree,r,28,event?.step.kind==='model'?event.label:null);
  writeCellValue(tree,r,29,(s.end_ns-s.start_ns)/1e6);
  writeCellValue(tree,r,30,s.kind==='model'?'Own call':descendants.length?'Subtree total':'No model charge');
  tree.getRange(`A${r}:I${r}`).format.fill=s.kind==='model'?'#f0f5fc':t.depth===0?'#e8eef7':'#ffffff';
  tree.getRange(`A${r}:${columnLetter(treeHeaders.length-1)}${r}`).format.rowHeight=54;
}
formatTreeAmounts(tree,treeLast);
tree.getRange(`A1:A${treeLast}`).format.columnWidth=48;
tree.getRange(`B1:C${treeLast}`).format.columnWidth=12;
tree.getRange(`D1:D${treeLast}`).format.columnWidth=26;
tree.getRange(`G1:G${treeLast}`).format.columnWidth=62;
tree.getRange(`A2:A${treeLast}`).format.wrapText=true;
tree.getRange(`G2:G${treeLast}`).format.wrapText=true;
tree.getRange(`H1:I${treeLast}`).format.columnWidth=38;
tree.getRange(`A2:${columnLetter(treeHeaders.length-1)}${treeLast}`).conditionalFormats.addCustom('$E2="error"',{fill:'#ffd3d3',font:{color:'#8b1010'}});
tree.getRange(`AD2:AD${treeLast}`).setNumberFormat('0.00');


// Check accounting parity and a changed execution forecast before writing the
// workbook. Restore the requested 100,000 default after the recalculation probe.
wb.recalculate();
const expected=Number(data.expected_usd);
const actual=workbookCell(turns,2,5).values[0][0];
// Permit floating-point noise only; a larger difference means workbook formulas
// disagree with the authoritative Python subtotal and export must stop.
if(Math.abs(actual-expected)>1e-10) throw new Error(`USD mismatch ${actual} vs ${expected}`);
const initial=workbookCell(turns,2,6).values[0][0];
writeCellValue(ref,2,1,200000);wb.recalculate();
const changed=workbookCell(turns,2,6).values[0][0];
const expectedChanged=Math.round(expected*Number(data.prices.exchange.rate)*200000);
// After changing executions, rounded EUR must match the independently scaled
// expectation; otherwise the workbook's editable forecast is not trustworthy.
if(Math.abs(changed-expectedChanged)>0.00001) throw new Error('Execution multiplier does not recalculate');
writeCellValue(ref,2,1,100000);wb.recalculate();
console.log(JSON.stringify({runUSD:actual,projectedEUR:initial,projectionAt200000:changed,rows:last,treeRows:treeLast}));
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:10},maxChars:1500})).ndjson);
console.log((await wb.inspect({kind:'table',range:'Turns!E1:H5',include:'values,formulas',tableMaxRows:5,tableMaxCols:4,maxChars:2200})).ndjson);
await fs.mkdir(path.dirname(output),{recursive:true});
// Preview images make wide-sheet layout review possible without opening Excel.
for(const [name,range,file] of [['Turns','A1:H22','turns'],['Turns','C7:I22','content'],['Execution tree','A1:G9','tree'],['Reference data','A1:D8','reference']]){
  const image=await wb.render({sheetName:name,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(path.dirname(output),`${file}-preview.png`),new Uint8Array(await image.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(output);
console.log(output);
