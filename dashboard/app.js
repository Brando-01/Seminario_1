"use strict";

const $ = (id) => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
const state = {dataset:"ieee", overview:null, graph:null, caseIndex:0, step:0, timer:null, runId:null};
const colors = {baseline:"#b2764f",autoencoder:"#8b6d92",gnn:"#4775a1",hybrid:"#2d746e"};
const labels = {baseline:"Baseline",autoencoder:"Autoencoder",gnn:"GraphSAGE",hybrid:"Híbrido GNN–AE"};
const descriptions = {
  ieee:"IEEE-CIS: transacciones relacionadas por proxies de tarjeta, tarjeta/correo o dispositivo. No son cuentas identificadas con certeza.",
  paysim:"PaySim: transacciones relacionadas cuando comparten una cuenta de origen o destino. Cada flecha respeta el orden temporal."
};
const fmtInt = (value) => Number(value || 0).toLocaleString("es-ES");
const fixed = (value) => Number(value || 0).toLocaleString("es-ES",{minimumFractionDigits:2,maximumFractionDigits:2});
const pct = (value) => `${fixed(100 * Number(value || 0))} %`;
const num = (value) => fixed(value);

async function getJSON(url){
  const response = await fetch(url,{cache:"no-store"});
  const data = await response.json();
  if(!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function element(tag,attrs={}){
  const node=document.createElementNS(svgNS,tag);
  Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,String(value)));
  return node;
}
function clearTimer(){if(state.timer){clearInterval(state.timer);state.timer=null;}$("playButton").textContent="▶ Ver formación";}

async function refreshOverview(){
  state.overview=await getJSON("/api/overview");
  const complete=state.overview.runs.filter(run=>run.complete).length;
  const partial=state.overview.runs.length-complete;
  $("runStatus").textContent=`${complete} corridas completas${partial?` · ${partial} en progreso`:""}`;
  renderRunOptions();
}
function renderRunOptions(){
  const select=$("runSelect");
  const available=state.overview.runs.filter(run=>run.ready_datasets.includes(state.dataset));
  const former=state.runId;
  select.replaceChildren();
  for(const run of available){
    const option=document.createElement("option");option.value=run.id;
    const date=run.id.slice(-15).replace("_"," · ");
    const title=run.kind==="thesis"?"Protocolo congelado · 4 modelos y SHAP":run.kind==="comparison"?"Comparación de 4 modelos":run.kind==="minibatch"?"Mejora por lotes · híbrido y baseline":run.kind==="feature_selection"?"Selección de atributos · híbrido":"Exploración híbrida";
    option.textContent=`${title} · ${date}${run.complete?"":" · parcial"}`;
    select.append(option);
  }
  state.runId=available.some(run=>run.id===former)?former:(available[0]?.id||null);
  if(state.runId)select.value=state.runId;
  if(!state.runId){$("runIntro").textContent="Todavía no hay una corrida completa para este dataset.";$("metricsBody").replaceChildren();}
}

async function selectDataset(dataset){
  clearTimer();state.dataset=dataset;state.caseIndex=0;state.step=0;
  $("metricsBody").replaceChildren();$("runIntro").textContent="Cargando el escenario seleccionado…";
  $("shapDetails").hidden=true;$("comparisonFinding").textContent="";
  document.querySelectorAll(".dataset-button").forEach(button=>button.classList.toggle("active",button.dataset.dataset===dataset));
  $("datasetExplanation").textContent=descriptions[dataset];
  try{const graph=await getJSON(`/api/graph/${dataset}`);if(state.dataset!==dataset)return;state.graph=graph;renderGraphData();}
  catch(error){if(state.dataset!==dataset)return;state.graph=null;$("stats").textContent="No hay cache visual. Inicia el panel con los CSV presentes para prepararla.";$("graphSvg").replaceChildren();$("caseTitle").textContent="Ejemplo no disponible";$("graphNote").textContent=String(error.message);}
  renderRunOptions();
  await loadRun();
}

function renderGraphData(){
  const summary=state.graph.summary, splits=summary.splits;
  renderStats(splits,"ventana del ejemplo visual");
  $("graphNote").textContent=`Fuente visual: ${state.graph.source_run}, semilla ${state.graph.source_seed}. ${state.graph.note} Los casos se eligieron por puntajes extremos para explicar el recorrido; no son una muestra representativa.`;
  const buttons=$("caseButtons");buttons.replaceChildren();
  state.graph.cases.forEach((item,index)=>{
    const button=document.createElement("button");button.type="button";button.textContent=index===0?"Fraude":index===1?"Falsa alerta":"Legítima";
    button.addEventListener("click",()=>{state.caseIndex=index;state.step=0;clearTimer();renderCase();});buttons.append(button);
  });
  renderCase();
}
function renderStats(splits,source){
  const stats=[
    ["OPERACIONES USADAS",splits.train.rows+splits.validation.rows+splits.test.rows,source,""],
    ["ENTRENAMIENTO",splits.train.rows,`${fmtInt(splits.train.fraud)} fraudes · ${pct(splits.train.fraud_rate)}`,"train"],
    ["VALIDACIÓN",splits.validation.rows,`${fmtInt(splits.validation.fraud)} fraudes · ${pct(splits.validation.fraud_rate)}`,""],
    ["PRUEBA",splits.test.rows,`${fmtInt(splits.test.fraud)} fraudes · ${pct(splits.test.fraud_rate)}`,"test"]
  ];
  $("stats").replaceChildren(...stats.map(([name,count,note,cls])=>{
    const card=document.createElement("div");card.className=`stat-card ${cls}`;
    const small=document.createElement("small");small.textContent=name;
    const strong=document.createElement("strong");strong.textContent=fmtInt(count);
    const span=document.createElement("span");span.textContent=note;
    card.append(small,strong,span);return card;
  }));
}

function positionsFor(nodes){
  const compact=window.matchMedia("(max-width: 700px)").matches;
  const groups={historial:nodes.filter(node=>node.role==="historial"),direct:nodes.filter(node=>node.role==="vecino directo")};
  const position={};
  const directX=groups.historial.length?(compact?300:460):(compact?150:270);
  for(const [group,x] of [[groups.historial,compact?105:155],[groups.direct,directX]]){
    group.forEach((node,index)=>{position[node.id]={x,y:compact?45+(index+1)*(500/(group.length+1)):70+(index+1)*(300/(group.length+1))};});
  }
  const target=nodes.find(node=>node.role==="objetivo");if(target)position[target.id]={x:compact?490:745,y:compact?295:220};
  return position;
}
function renderCase(){
  if(!state.graph)return;
  const current=state.graph.cases[state.caseIndex];
  state.step=current.nodes.length;
  [...$("caseButtons").children].forEach((button,index)=>button.classList.toggle("active",index===state.caseIndex));
  $("caseTitle").textContent=current.title;
  const facts=[["ID anonimizado",current.target],["Etiqueta real",current.actual_fraud?"Fraude":"Legítima"],["Importe",current.amount==null?"N/D":fmtInt(current.amount)],["Aristas de entrada",current.direct_neighbors]];
  $("caseFacts").replaceChildren(...facts.map(([name,value])=>{
    const line=document.createElement("div");line.className="fact";const a=document.createElement("span");a.textContent=name;
    const b=document.createElement("strong");b.textContent=String(value);if(name==="Etiqueta real")b.className=current.actual_fraud?"fraud":"legit";
    line.append(a,b);return line;
  }));
  $("nodeDetail").textContent="Haz clic sobre un nodo para conocer su papel. Las aristas muestran relaciones reales, no similitudes inventadas.";
  drawGraph(current);
  document.dispatchEvent(new CustomEvent("fraudlab:case",{detail:{dataset:state.dataset,case:current,sourceRun:state.graph.source_run,sourceSeed:state.graph.source_seed}}));
}
function drawGraph(current){
  const svg=$("graphSvg");svg.replaceChildren();
  const compact=window.matchMedia("(max-width: 700px)").matches;
  svg.setAttribute("viewBox",compact?"0 0 560 590":"0 0 900 430");
  const defs=element("defs");const marker=element("marker",{id:"arrow",markerWidth:7,markerHeight:7,refX:6,refY:3.5,orient:"auto"});
  marker.append(element("path",{d:"M0,0 L7,3.5 L0,7 Z",fill:"#7d999a"}));defs.append(marker);svg.append(defs);
  svg.append(element("line",{x1:compact?35:50,y1:compact?550:374,x2:compact?525:860,y2:compact?550:374,class:"axis-line"}));
  const positions=positionsFor(current.nodes);
  for(const edge of current.edges){
    const a=positions[edge.source],b=positions[edge.target];if(!a||!b)continue;
    const bend=(b.x-a.x)*.43;
    const line=element("path",{d:`M${a.x+18},${a.y} C${a.x+bend},${a.y} ${b.x-bend},${b.y} ${b.x-18},${b.y}`,class:`graph-edge${edge.target===current.target?" to-target":""}`,"marker-end":"url(#arrow)","data-source":edge.source,"data-target":edge.target});
    const title=element("title");title.textContent=edge.relation;line.append(title);svg.append(line);
  }
  for(const node of current.nodes){
    const pos=positions[node.id];if(!pos)continue;
    const group=element("g",{class:`graph-node ${node.role==="objetivo"?"target":node.role==="vecino directo"?"direct":"history"}${node.role==="objetivo"&&!current.actual_fraud?" legitimate":""}`,"data-id":node.id,tabindex:0,role:"button","aria-label":`${node.id}, ${node.role}, ${node.time}`});
    group.append(element("circle",{cx:pos.x,cy:pos.y,r:node.role==="objetivo"?31:24}));
    const label=element("text",{x:pos.x,y:pos.y+4});label.textContent=node.id;group.append(label);
    const time=element("text",{x:pos.x,y:pos.y+46,class:"graph-label"});time.textContent=node.time;group.append(time);
    const describe=()=>{$("nodeDetail").textContent=`${node.id} · ${node.role} · ${node.time} · partición: ${node.partition}. ${node.role==="objetivo"?"Es la transacción cuya predicción estamos observando.":"Aporta contexto histórico; su etiqueta no se usa para inferir el objetivo."}`;};
    group.addEventListener("click",describe);group.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();describe();}});svg.append(group);
  }
  updateGraphStep();
}
function updateGraphStep(){
  if(!state.graph)return;
  const nodes=state.graph.cases[state.caseIndex].nodes;
  const shown=new Set(nodes.slice(0,state.step).map(node=>node.id));
  $("graphSvg").querySelectorAll(".graph-node").forEach(node=>node.classList.toggle("visible",shown.has(node.dataset.id)));
  $("graphSvg").querySelectorAll(".graph-edge").forEach(edge=>edge.classList.toggle("visible",shown.has(edge.dataset.source)&&shown.has(edge.dataset.target)));
  $("stepLabel").textContent=`${state.step} / ${nodes.length} transacciones`;
  $("progressFill").style.width=`${nodes.length?100*state.step/nodes.length:0}%`;
}
function nextStep(){if(!state.graph)return;const total=state.graph.cases[state.caseIndex].nodes.length;if(state.step<total)state.step++;else{clearTimer();return;}updateGraphStep();if(state.step===total)clearTimer();}
function play(){if(!state.graph)return;if(state.timer){clearTimer();return;}if(state.step>=state.graph.cases[state.caseIndex].nodes.length){state.step=0;updateGraphStep();}$("playButton").textContent="❚❚ Pausar";requestAnimationFrame(()=>requestAnimationFrame(nextStep));state.timer=setInterval(nextStep,820);}

function mean(rows,key,source="metrics") {return rows.reduce((sum,row)=>sum+Number(row[source]?.[key]||0),0)/Math.max(rows.length,1);}
async function loadRun(){
  if(!state.runId)return;
  const runId=state.runId,dataset=state.dataset;
  $("runIntro").textContent="Cargando resultados guardados…";$("metricsBody").replaceChildren();
  $("comparisonFinding").textContent="";$("shapDetails").hidden=true;
  try{const detail=await getJSON(`/api/run/${runId}/${dataset}`);if(runId!==state.runId||dataset!==state.dataset)return;renderRun(detail);}
  catch(error){if(runId!==state.runId||dataset!==state.dataset)return;$("runIntro").textContent=`No se pudo leer esta corrida: ${error.message}`;$("metricsBody").replaceChildren();}
}
function renderRun(detail){
  const current=state.overview.runs.find(run=>run.id===detail.id),isHybrid=detail.kind==="hybrid";
  $("runBadge").textContent=current?.complete?"CORRIDA COMPLETA":"RESULTADO PARCIAL";
  $("runBadge").classList.toggle("partial",!current?.complete);
  const summary=detail.summary?.splits;
  if(summary)renderStats(summary,`datos de ${detail.id}`);
  const runDescription=detail.kind==="feature_selection"
    ?`Exploración del híbrido con ${detail.feature_count} atributos elegidos usando solo entrenamiento.`
    :detail.kind==="thesis"?`Protocolo congelado de cuatro modelos; ${detail.feature_count} atributos comunes. ${detail.shap?"SHAP calculado del híbrido con contexto fijo.":"SHAP aún no generado en esta corrida."}`
    :detail.kind==="minibatch"?`Híbrido entrenado por lotes frente al baseline. ${detail.feature_set==="enriched"?"Preprocesamiento enriquecido":"Variables originales"}; ${detail.feature_count} atributos.`
    :isHybrid?"Exploración del híbrido: variante elegida por AP de validación.":"Comparación de cuatro modelos sobre el mismo test.";
  $("runIntro").textContent=`${runDescription} ${summary?`${fmtInt(summary.test.rows)} transacciones de prueba; ${fmtInt(summary.test.fraud)} fraudes.`:""}`;
  const grouped=new Map();
  for(const row of detail.rows){if(!grouped.has(row.model))grouped.set(row.model,[]);grouped.get(row.model).push(row);}
  const winner=[...grouped].sort((a,b)=>mean(b[1],"auc_pr")-mean(a[1],"auc_pr"))[0]?.[0];
  const body=$("metricsBody");body.replaceChildren();
  for(const [model,rows] of grouped){
    const tr=document.createElement("tr");if(model===winner)tr.className="winner";
    const values=[labels[model]||model,pct(mean(rows,"auc_pr")),pct(mean(rows,"precision")),pct(mean(rows,"recall")),pct(mean(rows,"f1")),pct(mean(rows,"recall","operating_metrics")),num(mean(rows,"false_positives_per_1000","operating_metrics"))];
    const columns=["Modelo","AP","Precision","Recall","F1","Recall operativo","FP / 1000"];
    values.forEach((value,index)=>{const td=document.createElement("td");td.dataset.label=columns[index];td.textContent=value;if(index===1){const bar=document.createElement("div");bar.className="metric-bar";const fill=document.createElement("span");fill.style.width=`${Math.max(1,100*mean(rows,"auc_pr"))}%`;bar.append(fill);td.append(bar);}tr.append(td);});
    body.append(tr);
  }
  const baselineRows=grouped.get("baseline"),hybridRows=grouped.get("hybrid");
  const finding=$("comparisonFinding");
  if(baselineRows&&hybridRows){
    const difference=100*(mean(hybridRows,"auc_pr")-mean(baselineRows,"auc_pr"));
    const baselineF1=pct(mean(baselineRows,"f1")),hybridF1=pct(mean(hybridRows,"f1"));
    finding.textContent=difference>=0
      ?`En esta corrida, el híbrido supera al baseline por ${num(difference)} puntos de AP. F1: híbrido ${hybridF1}, baseline ${baselineF1}. Hace falta confirmarlo en otros periodos.`
      :`En esta corrida, el baseline supera al híbrido por ${num(-difference)} puntos de AP. F1: baseline ${baselineF1}, híbrido ${hybridF1}. La arquitectura híbrida aún no demuestra ventaja.`;
  }else{
    finding.textContent="Esta corrida incluye solo el híbrido. No permite afirmar que supera al baseline ni a los otros componentes.";
  }
  const candidateBox=$("candidateBox");candidateBox.hidden=!isHybrid;
  if(isHybrid){
    const byCandidate=new Map();for(const row of detail.candidates){if(!byCandidate.has(row.candidate))byCandidate.set(row.candidate,[]);byCandidate.get(row.candidate).push(row.validation_ap);}
    const selected=detail.rows[0]?.candidate||"N/D";
    candidateBox.textContent=`Variantes en validación: ${[...byCandidate].map(([name,values])=>`${name}: ${pct(values.reduce((a,b)=>a+b,0)/values.length)}`).join(" · ")}. Elegida: ${selected}. Esto no prueba superioridad frente al baseline.`;
  }
  $("resultCaveat").textContent=detail.caveat+(state.dataset==="paysim"&&detail.kind!=="thesis"?" PaySim tiene solo 36 fraudes en el test histórico; revisa cada semilla.":"");
  $("curveSeed").textContent=`Semilla ${detail.curve_seed} · test`;
  renderCurve(detail.curves);
  renderSeedRows(detail);
  renderShap(detail);
}
function renderShap(detail){
  const box=$("shapDetails"),content=$("shapContent");content.replaceChildren();box.hidden=!detail.shap;
  if(!detail.shap)return;
  const note=document.createElement("p");note.textContent=`SHAP del híbrido real: atributos agrupados de la transacción con vecinos y aristas fijos. Ranking de ${detail.shap.random_test_sample.length} transacciones aleatorias de test, no de toda la población; los casos adicionales no se agregan al ranking. No implica causalidad y las perturbaciones no garantizan combinaciones económicamente plausibles. Error de aditividad máximo: ${detail.shap.max_additivity_error.toExponential(2)}.`;
  content.append(note);
  const global=document.createElement("img");global.src=`/api/shap/${detail.id}/${detail.dataset}/global_importance.png`;global.alt="Importancia SHAP agrupada en la muestra aleatoria";global.loading="lazy";content.append(global);
  for(const [kind,index] of Object.entries(detail.shap.illustrative_local_cases)){
    const names={TP:"Fraude detectado",FP:"Falsa alerta",FN:"Fraude no detectado",TN:"Operación legítima sin alerta"};
    const title=document.createElement("h3");title.textContent=`${names[kind]} (${kind}) · índice ${index} en test`;
    const local=document.createElement("img");local.src=`/api/shap/${detail.id}/${detail.dataset}/local_${kind}.png`;local.alt=`Contribuciones SHAP del caso ${kind}`;local.loading="lazy";content.append(title,local);
  }
}
function renderCurve(curves){
  const svg=$("prChart");svg.replaceChildren();const left=55,right=575,top=18,bottom=253;
  for(let i=0;i<=4;i++){
    const y=bottom-(bottom-top)*i/4;
    svg.append(element("line",{x1:left,y1:y,x2:right,y2:y,stroke:"#d8ddd8","stroke-dasharray":i===0?"0":"3 6"}));
    const text=element("text",{x:46,y:y+4,fill:"#68777b","font-size":10,"text-anchor":"end"});text.textContent=`${i*25}`;svg.append(text);
  }
  svg.append(element("line",{x1:left,y1:bottom,x2:right,y2:bottom,stroke:"#a8b5b0"}));
  for(let i=0;i<=4;i++){const x=left+(right-left)*i/4;const text=element("text",{x,y:bottom+18,fill:"#68777b","font-size":10,"text-anchor":"middle"});text.textContent=`${i*25}`;svg.append(text);}
  const axisX=element("text",{x:(left+right)/2,y:294,fill:"#53666a","font-size":11,"text-anchor":"middle"});axisX.textContent="Recall (%)";svg.append(axisX);
  const legend=$("curveLegend");legend.replaceChildren();
  for(const [model,points] of Object.entries(curves)){
    if(!points?.length)continue;
    const path=element("polyline",{points:points.map(([recall,precision])=>`${left+recall*(right-left)},${bottom-precision*(bottom-top)}`).join(" "),fill:"none",stroke:colors[model]||"white","stroke-width":2.5,"stroke-linejoin":"round"});svg.append(path);
    const span=document.createElement("span"),dot=document.createElement("i");dot.className="legend-dot";dot.style.backgroundColor=colors[model]||"white";span.append(dot,document.createTextNode(labels[model]||model));legend.append(span);
  }
}
function renderSeedRows(detail){
  const parent=$("seedRows");parent.replaceChildren();
  for(const row of detail.rows){
    const line=document.createElement("div");line.className="seed-row";
    let seconds=row.seconds;
    if(seconds==null && detail.kind==="hybrid")seconds=detail.candidates.find(candidate=>candidate.seed===row.seed&&candidate.candidate===row.candidate)?.seconds;
    const timing=row.inference_seconds!=null?`train ${num(seconds)} s · infer ${num(row.inference_seconds)} s`:seconds==null?"tiempo N/D":`corrida ~${num(seconds)} s`;
    const values=[`${labels[row.model]||row.model} · semilla ${row.seed}`,`AP ${pct(row.metrics.auc_pr)}`,`F1 ${pct(row.metrics.f1)}`,`FP/1000 ${num(row.operating_metrics.false_positives_per_1000)}`,timing];
    values.forEach((value,index)=>{const item=document.createElement(index===0?"strong":"span");item.textContent=value;line.append(item);});parent.append(line);
  }
}

document.addEventListener("DOMContentLoaded",async()=>{
  document.querySelectorAll(".dataset-button").forEach(button=>button.addEventListener("click",()=>selectDataset(button.dataset.dataset)));
  $("playButton").addEventListener("click",play);$("nextButton").addEventListener("click",nextStep);
  $("runSelect").addEventListener("change",async(event)=>{state.runId=event.target.value;await loadRun();});
  try{await refreshOverview();await selectDataset("ieee");}
  catch(error){$("runStatus").textContent="No se pudo conectar con el servidor";$("stats").textContent=error.message;}
  setInterval(async()=>{try{await refreshOverview();}catch(_error){/* Keep the last visible result. */}},20000);
  let wasCompact=window.matchMedia("(max-width: 700px)").matches;
  window.addEventListener("resize",()=>{const compact=window.matchMedia("(max-width: 700px)").matches;if(compact!==wasCompact&&state.graph){wasCompact=compact;drawGraph(state.graph.cases[state.caseIndex]);}});
});
