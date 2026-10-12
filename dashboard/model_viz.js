"use strict";

/* Every diagram is explanatory. Only the figures marked "guardado" come from model output. */
const modelMeta = {
  baseline: {
    eyebrow: "01 / BASELINE TABULAR",
    title: "Árboles que combinan decisiones",
    description: "El modelo supervisado evalúa atributos de la transacción en varios árboles y reúne sus aportes en un puntaje.",
    steps: ["Entrada tabular", "Reglas de los árboles", "Aportes combinados", "Puntaje guardado"],
  },
  autoencoder: {
    eyebrow: "02 / AUTOENCODER",
    title: "Comprimir y reconstruir",
    description: "Aprende patrones normales: comprime los atributos, intenta reconstruirlos y mide la diferencia.",
    steps: ["Vector original", "Compresión", "Reconstrucción", "Error de reconstrucción"],
  },
  gnn: {
    eyebrow: "03 / GRAPHSAGE",
    title: "Agregar información del pasado",
    description: "Cada transacción recibe mensajes de vecinos anteriores relacionados y forma una representación para clasificarla.",
    steps: ["Vecinos anteriores", "Agregación", "Representación", "Puntaje guardado"],
  },
  hybrid: {
    eyebrow: "04 / HÍBRIDO GNN–AE",
    title: "Fusionar tres representaciones",
    description: "Combina la representación tabular, el autoencoder y GraphSAGE antes de un clasificador aprendido; no promedia sus puntajes.",
    steps: ["Tres ramas", "Representaciones", "Concatenación", "Puntaje guardado"],
  },
};

const scenes = {
  baseline: `
    <div class="viz-flow">
      <div class="viz-block viz-input"><span class="viz-number">01 / ENTRADA</span><strong>Transacción</strong><span data-bind="featureCount"></span><div class="feature-lines"><i></i><i></i><i></i><i></i></div><small data-bind="amount"></small></div>
      <div class="viz-wire"></div>
      <div class="viz-block viz-process"><span class="viz-number">02 / APRENDIZAJE</span><strong>Ensamble de árboles</strong><div class="tree-row"><span class="tree-shape"><i></i><i></i></span><span class="tree-shape"><i></i><i></i></span><span class="tree-shape"><i></i><i></i></span></div><small>Ramas ilustrativas; no son rutas internas registradas.</small></div>
      <div class="viz-wire"></div>
      <div class="viz-block viz-output"><span class="viz-number">03 / SALIDA</span><strong>Puntaje de riesgo</strong><b data-bind="baselineScore"></b><small>Escala 0–100 % · no calibrada</small></div>
    </div>`,
  autoencoder: `
    <div class="viz-flow ae-flow">
      <div class="viz-block viz-input"><span class="viz-number">01 / ENTRADA</span><strong>Vector original</strong><span data-bind="featureCount"></span><div class="signal-bars"><i></i><i></i><i></i><i></i><i></i><i></i></div><small data-bind="amount"></small></div>
      <div class="viz-wire"></div>
      <div class="viz-block viz-process"><span class="viz-number">02 / CUELLO DE BOTELLA</span><strong>Compresión</strong><div class="latent-orbit"><span data-bind="latentCount"></span></div><small>Espacio latente del modelo</small></div>
      <div class="viz-wire"></div>
      <div class="viz-block viz-output"><span class="viz-number">03 / RECONSTRUCCIÓN</span><strong>Comparar con la entrada</strong><div class="signal-bars reconstructed"><i></i><i></i><i></i><i></i><i></i><i></i></div><b data-bind="aeScore"></b><small>Error real · percentil <span data-bind="aePercentile"></span></small></div>
    </div>`,
  gnn: `
    <div class="viz-flow gnn-flow">
      <div class="viz-block viz-input"><span class="viz-number">01 / HISTORIAL</span><strong>Vecinos anteriores</strong><div class="neighbor-list" data-bind="neighborList"></div><small data-bind="neighborCount"></small></div>
      <div class="viz-wire"></div>
      <div class="viz-block viz-process"><span class="viz-number">02 / GRAPHSAGE</span><strong>Agregación de mensajes</strong><div class="aggregate-glyph"><i></i><i></i><i></i><span>Σ</span></div><small>Los vecinos aportan contexto al objetivo.</small></div>
      <div class="viz-wire"></div>
      <div class="viz-block viz-output"><span class="viz-number">03 / SALIDA</span><strong>Puntaje de riesgo</strong><b data-bind="gnnScore"></b><small>Escala 0–100 % · no calibrada</small></div>
    </div>`,
  hybrid: `
    <div class="hybrid-flow"><div class="hybrid-branches">
      <div class="hybrid-branch"><span>01</span><strong>Transacción tabular</strong><small data-bind="featureCount"></small><i></i></div>
      <div class="hybrid-branch"><span>02</span><strong>Autoencoder</strong><small>Latente + error</small><i></i></div>
      <div class="hybrid-branch"><span>03</span><strong>GraphSAGE</strong><small data-bind="neighborCount"></small><i></i></div>
    </div><div class="hybrid-merge"><span>CONCATENACIÓN</span><strong>Representación fusionada</strong><small>Clasificador entrenado con etiquetas</small></div><div class="hybrid-result"><span>PUNTAJE GUARDADO</span><b data-bind="hybridScore"></b><small>Riesgo en escala 0–100 % · no calibrado</small></div></div>`,
};

let selectedModel = "baseline";
let caseData = null;
let replayTimer = null;
let stepTimers = [];
const scene = document.getElementById("modelScene");
const number = value => Number(value || 0).toLocaleString("es-ES", {minimumFractionDigits:2, maximumFractionDigits:2});
const percent = value => `${number(100 * Number(value || 0))} %`;

function bindScene() {
  if (!caseData) return;
  const example = caseData.case;
  const values = {
    featureCount: `${example.feature_count} atributos`,
    amount: example.amount == null ? "Importe no disponible" : `Importe: ${Number(example.amount).toLocaleString("es-ES", {maximumFractionDigits:3})}`,
    latentCount: `${example.latent_dimension} dimensiones`,
    baselineScore: percent(example.model_scores.baseline),
    aeScore: number(example.model_scores.autoencoder),
    aePercentile: `${number(example.ae_percentile)} %`,
    neighborCount: `${example.direct_neighbors} vecinos directos`,
    gnnScore: percent(example.model_scores.gnn),
    hybridScore: percent(example.model_scores.hybrid),
  };
  scene.querySelectorAll("[data-bind]").forEach(node => {
    if (node.dataset.bind === "neighborList") {
      const neighbors = example.nodes.filter(item => item.role === "vecino directo").slice(0, 3);
      node.replaceChildren(...neighbors.map(item => {
        const chip = document.createElement("span");
        chip.className = "neighbor-chip";
        chip.textContent = item.id;
        return chip;
      }));
      if (example.direct_neighbors > 3) {
        const extra = document.createElement("span");
        extra.className = "neighbor-chip extra";
        extra.textContent = `+${example.direct_neighbors - 3}`;
        node.append(extra);
      }
    } else {
      node.textContent = values[node.dataset.bind] || "—";
    }
  });
  document.getElementById("modelProvenance").textContent = `Caso demostrativo de puntaje extremo · puntaje real: ${caseData.sourceRun}, semilla ${caseData.sourceSeed}`;
}

function stopReplay() {
  if (replayTimer) clearTimeout(replayTimer);
  stepTimers.forEach(clearTimeout);
  replayTimer = null;
  stepTimers = [];
  scene.classList.remove("is-playing");
}

function replay() {
  stopReplay();
  if (!caseData) return;
  void scene.offsetWidth;
  scene.classList.add("is-playing");
  const descriptions = modelMeta[selectedModel].steps;
  document.getElementById("modelStep").textContent = `1 / 4 · ${descriptions[0]}`;
  for (let index = 1; index < descriptions.length; index++) {
    stepTimers.push(setTimeout(() => {
      document.getElementById("modelStep").textContent = `${index + 1} / 4 · ${descriptions[index]}`;
    }, index * 900));
  }
  replayTimer = setTimeout(() => { scene.classList.remove("is-playing"); replayTimer = null; }, 4300);
}

function selectModel(model) {
  if (!modelMeta[model]) return;
  stopReplay();
  selectedModel = model;
  document.querySelectorAll(".model-tab").forEach(tab => {
    const active = tab.dataset.model === model;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  const meta = modelMeta[model];
  document.getElementById("modelEyebrow").textContent = meta.eyebrow;
  document.getElementById("modelTitle").textContent = meta.title;
  document.getElementById("modelDescription").textContent = meta.description;
  scene.innerHTML = scenes[model];
  scene.dataset.model = model;
  bindScene();
  replay();
}

document.querySelectorAll(".model-tab").forEach(tab => tab.addEventListener("click", () => selectModel(tab.dataset.model)));
document.getElementById("modelReplay").addEventListener("click", replay);
document.addEventListener("fraudlab:case", event => { caseData = event.detail; selectModel(selectedModel); });
selectModel("baseline");
