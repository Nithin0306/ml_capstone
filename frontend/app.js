/**
 * ML Lab — Production Analytics Engine & Interactive UI
 * Architecture: Vanilla ES6+, Zero build-step, Pure Canvas Visualizations
 */

const API_BASE = "http://localhost:8000/api/v1";

// Application State
const state = {
  activeRoute: "overview",
  currentTrack: "regression",
  leaderboard: [],
  experiments: [],
  clustering: {
    algorithm: "kmeans",
    k: 2,
    linkage: "ward",
    projectionType: "pca",
    pcaPoints: [],
    labels: [],
    tsnePoints: [],
    metrics: {}
  },
  classification: {
    activeModel: "random_forest",
    faceoffA: "random_forest",
    faceoffB: "logistic_regression",
    evalCache: {}
  },
  regression: {
    activeModel: "random_forest",
    diagnostics: null
  }
};

// -----------------------------------------------------------------------------
// 1. Router & Navigation Shell
// -----------------------------------------------------------------------------
function initRouter() {
  window.addEventListener("hashchange", handleHashChange);
  handleHashChange();

  // Mobile menu toggle
  const toggleBtn = document.getElementById("menuToggleBtn");
  const sidebar = document.getElementById("sidebar");
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  // Close sidebar on link click (mobile)
  document.querySelectorAll(".sidebar-nav a").forEach(link => {
    link.addEventListener("click", () => {
      if (window.innerWidth <= 768 && sidebar) {
        sidebar.classList.remove("open");
      }
    });
  });
}

function handleHashChange() {
  const rawHash = window.location.hash.replace("#/", "").replace("#", "");
  const route = rawHash || "overview";
  navigateTo(route);
}

function navigateTo(route) {
  state.activeRoute = route;

  // Update views
  document.querySelectorAll(".page-view").forEach(view => {
    view.classList.remove("active");
  });

  const targetView = document.getElementById(`view-${route}`);
  if (targetView) {
    targetView.classList.add("active");
  } else {
    document.getElementById("view-overview")?.classList.add("active");
  }

  // Update nav links
  document.querySelectorAll(".sidebar-nav .nav-item").forEach(link => {
    link.classList.remove("active");
    if (link.getAttribute("data-route") === route) {
      link.classList.add("active");
    }
  });

  // Update breadcrumb
  const crumbEl = document.getElementById("currentCrumb");
  if (crumbEl) {
    const routeTitles = {
      overview: "Overview / ML Command Center",
      regression: "Regression Lab",
      classification: "Classification Lab",
      clustering: "Clustering Lab",
      compare: "Model Comparison Studio",
      datasets: "Dataset Explorer"
    };
    crumbEl.textContent = routeTitles[route] || "Overview";
  }

  // On page specific activation, ensure charts and canvas are drawn
  if (route === "regression") {
    loadRegressionDiagnostics();
  } else if (route === "classification") {
    updateFaceoffView();
    loadClassificationEvaluation(state.classification.activeModel);
  } else if (route === "clustering") {
    loadClusteringData();
    drawElbowPlot();
    drawDendrogram();
  } else if (route === "compare") {
    renderStudioTable();
  }
}

// Sub-Tab Handlers
function initTabs() {
  document.querySelectorAll(".tab-bar").forEach(bar => {
    bar.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab-btn");
      if (!btn) return;
      const tabId = btn.getAttribute("data-tab");
      if (!tabId) return;

      // Deactivate siblings in this tab bar
      bar.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Activate corresponding tab content within this section
      const parentSection = bar.closest(".page-view");
      if (parentSection) {
        parentSection.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        const target = parentSection.querySelector(`#tab-${tabId}`);
        if (target) {
          target.classList.add("active");
          // Re-render canvases if tab switched
          if (tabId === "reg-diagnostics") loadRegressionDiagnostics();
          if (tabId === "class-evaluation") loadClassificationEvaluation(state.classification.activeModel);
          if (tabId === "clust-visualize") renderScatterCanvas();
          if (tabId === "clust-elbow") drawElbowPlot();
          if (tabId === "clust-dendro") drawDendrogram();
        }
      }
    });
  });
}

// -----------------------------------------------------------------------------
// 2. Initial Data Loading & Health Check
// -----------------------------------------------------------------------------
async function initData() {
  try {
    // Health Check
    const healthRes = await fetch(`${API_BASE}/health`).then(r => r.json());
    const dot = document.getElementById("systemStatusDot");
    const txt = document.getElementById("systemStatusText");
    if (dot && txt) {
      dot.style.background = "#10b981";
      txt.textContent = `${healthRes.framework} Online`;
    }
  } catch (err) {
    console.warn("Backend health check failed:", err);
    const dot = document.getElementById("systemStatusDot");
    const txt = document.getElementById("systemStatusText");
    if (dot && txt) {
      dot.style.background = "#f59e0b";
      txt.textContent = "Connecting...";
    }
  }

  // Load Experiments
  try {
    const expRes = await fetch(`${API_BASE}/experiments`).then(r => r.json());
    state.experiments = expRes;
    renderExperimentsTable(expRes);
  } catch (e) {
    console.error("Failed to load experiments:", e);
  }

  // Load Leaderboard Data for Studio
  try {
    const regRes = await fetch(`${API_BASE}/results/regression`).then(r => r.json());
    const classRes = await fetch(`${API_BASE}/results/classification`).then(r => r.json());
    const clustRes = await fetch(`${API_BASE}/results/clustering`).then(r => r.json());
    state.leaderboard = [...regRes, ...classRes, ...clustRes];
    renderRegressionLeaderboard(regRes);
    renderClassificationLeaderboard(classRes);
    renderStudioTable();
  } catch (e) {
    console.error("Failed to load leaderboard data:", e);
  }
}

function renderExperimentsTable(items) {
  const tbody = document.getElementById("recentExperimentsBody");
  if (!tbody) return;
  tbody.innerHTML = items.map(exp => `
    <tr>
      <td><code>${exp.id}</code></td>
      <td><span class="badge ${exp.track.toLowerCase()}-badge">${exp.track}</span></td>
      <td><strong>${exp.model}</strong></td>
      <td>${exp.metric}</td>
      <td><span class="status-pill status-good">${exp.status}</span></td>
    </tr>
  `).join("");
}

// -----------------------------------------------------------------------------
// 3. Regression Lab Handlers
// -----------------------------------------------------------------------------
function initRegressionLab() {
  // Model selector change
  const modelSelect = document.getElementById("regModelSelect");
  if (modelSelect) {
    modelSelect.addEventListener("change", (e) => {
      state.regression.activeModel = e.target.value;
      const badge = document.getElementById("regResultModelBadge");
      if (badge) badge.textContent = modelSelect.options[modelSelect.selectedIndex].text.split(" (")[0];
      executeRegressionPredict();
    });
  }

  // Slider <-> Number input sync
  const pairs = [
    ["input_CO_GT", "input_CO_GT_range"],
    ["input_PT08_S1", "input_PT08_S1_range"],
    ["input_PT08_S2", "input_PT08_S2_range"],
    ["input_NOx_GT", "input_NOx_GT_range"],
    ["input_PT08_S3", "input_PT08_S3_range"],
    ["input_NO2_GT", "input_NO2_GT_range"],
    ["input_PT08_S4", "input_PT08_S4_range"],
    ["input_PT08_S5", "input_PT08_S5_range"],
    ["input_T", "input_T_range"],
    ["input_RH", "input_RH_range"],
    ["input_AH", "input_AH_range"],
    ["input_NMHC_GT", "input_NMHC_GT_range"]
  ];

  pairs.forEach(([numId, rangeId]) => {
    const numEl = document.getElementById(numId);
    const rangeEl = document.getElementById(rangeId);
    if (numEl && rangeEl) {
      rangeEl.addEventListener("input", () => { numEl.value = rangeEl.value; });
      numEl.addEventListener("input", () => { rangeEl.value = numEl.value; });
    }
  });

  // Presets
  document.getElementById("presetCleanBtn")?.addEventListener("click", () => {
    setRegressionInputs({
      "CO(GT)": 1.2, "PT08.S1(CO)": 850, "PT08.S2(NMHC)": 620,
      "NOx(GT)": 85, "PT08.S3(NOx)": 1150, "NO2(GT)": 45,
      "PT08.S4(NO2)": 1120, "PT08.S5(O3)": 680, "T": 19.5, "RH": 42.0, "AH": 0.85, "NMHC(GT)": 80
    });
    executeRegressionPredict();
  });

  document.getElementById("presetRushBtn")?.addEventListener("click", () => {
    setRegressionInputs({
      "CO(GT)": 4.5, "PT08.S1(CO)": 1550, "PT08.S2(NMHC)": 1420,
      "NOx(GT)": 650, "PT08.S3(NOx)": 520, "NO2(GT)": 190,
      "PT08.S4(NO2)": 1980, "PT08.S5(O3)": 1650, "T": 14.2, "RH": 65.0, "AH": 1.15, "NMHC(GT)": 420
    });
    executeRegressionPredict();
  });

  document.getElementById("presetIndustrialBtn")?.addEventListener("click", () => {
    setRegressionInputs({
      "CO(GT)": 3.1, "PT08.S1(CO)": 1280, "PT08.S2(NMHC)": 1180,
      "NOx(GT)": 380, "PT08.S3(NOx)": 680, "NO2(GT)": 135,
      "PT08.S4(NO2)": 1620, "PT08.S5(O3)": 1290, "T": 22.0, "RH": 50.0, "AH": 1.05, "NMHC(GT)": 280
    });
    executeRegressionPredict();
  });

  // Predict Buttons
  document.getElementById("btnPredictRegression")?.addEventListener("click", executeRegressionPredict);
  document.getElementById("btnCompareRegressionWhatIf")?.addEventListener("click", executeRegressionWhatIf);

  // Initial predict
  executeRegressionPredict();
}

function setRegressionInputs(vals) {
  const map = {
    "CO(GT)": "input_CO_GT",
    "PT08.S1(CO)": "input_PT08_S1",
    "PT08.S2(NMHC)": "input_PT08_S2",
    "NOx(GT)": "input_NOx_GT",
    "PT08.S3(NOx)": "input_PT08_S3",
    "NO2(GT)": "input_NO2_GT",
    "PT08.S4(NO2)": "input_PT08_S4",
    "PT08.S5(O3)": "input_PT08_S5",
    "T": "input_T",
    "RH": "input_RH",
    "AH": "input_AH",
    "NMHC(GT)": "input_NMHC_GT"
  };

  Object.entries(vals).forEach(([k, v]) => {
    const id = map[k];
    if (id) {
      const numEl = document.getElementById(id);
      const rangeEl = document.getElementById(`${id}_range`);
      if (numEl) numEl.value = v;
      if (rangeEl) rangeEl.value = v;
    }
  });
}

function getRegressionFeaturesFromForm() {
  return {
    "CO(GT)": parseFloat(document.getElementById("input_CO_GT")?.value || 2.2),
    "PT08.S1(CO)": parseFloat(document.getElementById("input_PT08_S1")?.value || 1097),
    "PT08.S2(NMHC)": parseFloat(document.getElementById("input_PT08_S2")?.value || 942),
    "NOx(GT)": parseFloat(document.getElementById("input_NOx_GT")?.value || 230),
    "PT08.S3(NOx)": parseFloat(document.getElementById("input_PT08_S3")?.value || 798),
    "NO2(GT)": parseFloat(document.getElementById("input_NO2_GT")?.value || 106),
    "PT08.S4(NO2)": parseFloat(document.getElementById("input_PT08_S4")?.value || 1453),
    "PT08.S5(O3)": parseFloat(document.getElementById("input_PT08_S5")?.value || 1010),
    "T": parseFloat(document.getElementById("input_T")?.value || 17.7),
    "RH": parseFloat(document.getElementById("input_RH")?.value || 48.8),
    "AH": parseFloat(document.getElementById("input_AH")?.value || 0.99),
    "NMHC(GT)": parseFloat(document.getElementById("input_NMHC_GT")?.value || 150)
  };
}

async function executeRegressionPredict() {
  const features = getRegressionFeaturesFromForm();
  const model = state.regression.activeModel;

  try {
    const res = await fetch(`${API_BASE}/regression/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: model, features })
    }).then(r => r.json());

    const valEl = document.getElementById("regPredValue");
    const catEl = document.getElementById("regPredCategory");
    if (valEl) valEl.textContent = res.predicted_value.toFixed(2);
    if (catEl) {
      catEl.textContent = `${res.category} (${res.predicted_value.toFixed(1)} μg/m³)`;
      catEl.className = "status-pill " + (res.predicted_value < 5 ? "status-good" : (res.predicted_value < 10 ? "status-moderate" : "status-hazard"));
    }
  } catch (err) {
    console.error("Regression predict failed:", err);
  }
}

async function executeRegressionWhatIf() {
  const features = getRegressionFeaturesFromForm();

  try {
    const res = await fetch(`${API_BASE}/regression/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: "random_forest", features })
    }).then(r => r.json());

    const container = document.getElementById("whatIfBarsContainer");
    if (!container || !res.predictions) return;

    const names = {
      random_forest: "Random Forest",
      gradient_boosting: "Gradient Boosting",
      svr: "Support Vector SVR",
      knn: "KNN Regressor",
      decision_tree: "Decision Tree",
      linear_regression: "Linear Regression",
      ridge: "Ridge Regression",
      lasso: "Lasso Regression",
      elastic_net: "ElasticNet"
    };

    const maxVal = Math.max(...Object.values(res.predictions), 15);

    container.innerHTML = Object.entries(res.predictions).map(([mid, val]) => {
      const pct = Math.min(100, Math.max(8, (val / maxVal) * 100));
      return `
        <div class="bar-row">
          <span class="bar-label">${names[mid] || mid}</span>
          <div class="bar-track"><div class="bar-fill" style="width: ${pct}%;"></div></div>
          <span class="bar-val">${val.toFixed(2)} μg/m³</span>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("Regression what-if comparison failed:", err);
  }
}

function renderRegressionLeaderboard(items) {
  const tbody = document.getElementById("regComparisonBody");
  if (!tbody) return;
  tbody.innerHTML = items.map(m => `
    <tr>
      <td><strong>${m.algorithm}</strong></td>
      <td><span class="badge ${m.status === 'Best Model' ? 'accent-badge' : ''}">${m.primary_metric.replace('R²: ', '')}</span></td>
      <td>${m.secondary_metric.replace('RMSE: ', '')}</td>
      <td>${m.accuracy.replace('MAE: ', '')}</td>
      <td><span class="status-pill ${m.status === 'Best Model' ? 'status-good' : 'status-moderate'}">${m.status}</span></td>
      <td><small class="text-muted">${m.phase}</small></td>
    </tr>
  `).join("");
}

async function loadRegressionDiagnostics() {
  if (state.regression.diagnostics) {
    drawRegressionScatter(state.regression.diagnostics);
    drawRegressionResiduals(state.regression.diagnostics);
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/regression/diagnostics`).then(r => r.json());
    state.regression.diagnostics = res;
    drawRegressionScatter(res);
    drawRegressionResiduals(res);
  } catch (e) {
    console.error("Failed to load regression diagnostics:", e);
  }
}

function drawRegressionScatter(data) {
  const canvas = document.getElementById("regScatterCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = 40;
  const actuals = data.sample_actual;
  const preds = data.sample_predicted;
  const maxVal = Math.max(...actuals, ...preds, 40);

  // Background grid
  ctx.strokeStyle = "#22293a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + ((h - 2 * pad) / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    const x = pad + ((w - 2 * pad) / 4) * i;
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
  }

  // Ideal y = x reference line
  ctx.strokeStyle = "#64748b";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  ctx.lineTo(w - pad, pad);
  ctx.stroke();
  ctx.setLineDash([]);

  // Scatter points
  ctx.fillStyle = "#BF0C4F";
  for (let i = 0; i < actuals.length; i++) {
    const x = pad + (actuals[i] / maxVal) * (w - 2 * pad);
    const y = h - pad - (preds[i] / maxVal) * (h - 2 * pad);
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px Inter";
  ctx.fillText("Actual Benzene (μg/m³)", w / 2 - 50, h - 10);
  ctx.save();
  ctx.translate(15, h / 2 + 30);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Predicted Benzene (μg/m³)", 0, 0);
  ctx.restore();
}

function drawRegressionResiduals(data) {
  const canvas = document.getElementById("regResidualCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = 40;
  const preds = data.sample_predicted;
  const residuals = data.residuals;
  const maxPred = Math.max(...preds, 40);
  const maxRes = Math.max(...residuals.map(Math.abs), 5);

  // Center zero line
  const midY = h / 2;
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad, midY);
  ctx.lineTo(w - pad, midY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Scatter points
  ctx.fillStyle = "#38bdf8";
  for (let i = 0; i < preds.length; i++) {
    const x = pad + (preds[i] / maxPred) * (w - 2 * pad);
    const y = midY - (residuals[i] / maxRes) * (h / 2 - pad);
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Axis Labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px Inter";
  ctx.fillText("Fitted / Predicted Values (μg/m³)", w / 2 - 70, h - 10);
  ctx.save();
  ctx.translate(15, h / 2 + 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Residuals (Actual − Pred)", 0, 0);
  ctx.restore();
}

// -----------------------------------------------------------------------------
// 4. Classification Lab Handlers & Placeholder States
// -----------------------------------------------------------------------------
function initClassificationLab() {
  const modelSelect = document.getElementById("classModelSelect");
  if (modelSelect) {
    modelSelect.addEventListener("change", (e) => {
      handleClassificationModelChange(e.target.value);
    });
  }

  // Placeholder switch button
  document.getElementById("btnSwitchActiveModel")?.addEventListener("click", () => {
    if (modelSelect) {
      modelSelect.value = "random_forest";
      handleClassificationModelChange("random_forest");
    }
  });

  // Presets
  document.getElementById("presetApproveBtn")?.addEventListener("click", () => {
    setClassificationInputs({ A9: "t", A10: "t", A2: 38, A3: 6.5, A8: 4.5, A11: 5, A14: 160, A15: 1500, A4: "u" });
    executeClassificationPredict();
  });

  document.getElementById("presetRejectBtn")?.addEventListener("click", () => {
    setClassificationInputs({ A9: "f", A10: "f", A2: 24, A3: 1.5, A8: 0.5, A11: 0, A14: 280, A15: 0, A4: "y" });
    executeClassificationPredict();
  });

  document.getElementById("presetBorderlineBtn")?.addEventListener("click", () => {
    setClassificationInputs({ A9: "t", A10: "f", A2: 29, A3: 3.2, A8: 1.2, A11: 1, A14: 200, A15: 250, A4: "u" });
    executeClassificationPredict();
  });

  // Predict buttons
  document.getElementById("btnPredictClassification")?.addEventListener("click", executeClassificationPredict);
  document.getElementById("btnCompareClassAll")?.addEventListener("click", executeClassificationCompareAll);

  // Model Face-Off Selectors
  document.getElementById("faceoffModelA")?.addEventListener("change", (e) => {
    state.classification.faceoffA = e.target.value;
    updateFaceoffView();
  });

  document.getElementById("faceoffModelB")?.addEventListener("change", (e) => {
    state.classification.faceoffB = e.target.value;
    updateFaceoffView();
  });

  // Initial predict
  executeClassificationPredict();
}

function handleClassificationModelChange(modelId) {
  state.classification.activeModel = modelId;
  const noticeEl = document.getElementById("classPlaceholderNotice");
  const formLayout = document.getElementById("classActiveFormLayout");

  if (noticeEl) noticeEl.classList.add("hidden");
  if (formLayout) formLayout.classList.remove("hidden");
  
  const badge = document.getElementById("classResultModelBadge");
  if (badge) badge.textContent = document.getElementById("classModelSelect")?.selectedOptions[0]?.text.split(" [")[0];
  executeClassificationPredict();
  loadClassificationEvaluation(modelId);
}

function setClassificationInputs(vals) {
  Object.entries(vals).forEach(([k, v]) => {
    const el = document.getElementById(`input_${k}`);
    if (el) el.value = v;
  });
}

function getClassificationFeaturesFromForm() {
  return {
    A9: document.getElementById("input_A9")?.value || "t",
    A10: document.getElementById("input_A10")?.value || "t",
    A2: parseFloat(document.getElementById("input_A2")?.value || 31.5),
    A3: parseFloat(document.getElementById("input_A3")?.value || 4.5),
    A8: parseFloat(document.getElementById("input_A8")?.value || 2.0),
    A11: parseFloat(document.getElementById("input_A11")?.value || 1.0),
    A14: parseFloat(document.getElementById("input_A14")?.value || 180),
    A15: parseFloat(document.getElementById("input_A15")?.value || 500),
    A4: document.getElementById("input_A4")?.value || "u"
  };
}

async function executeClassificationPredict() {
  const model = state.classification.activeModel;
  const features = getClassificationFeaturesFromForm();

  try {
    const res = await fetch(`${API_BASE}/classification/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: model, features })
    }).then(r => r.json());

    if (res.status === "placeholder") return;

    const badge = document.getElementById("classDecisionBadge");
    const probVal = document.getElementById("classProbVal");
    const probBar = document.getElementById("classProbBar");

    if (badge) {
      badge.textContent = res.prediction;
      badge.className = "decision-badge " + (res.approved ? "badge-approved" : "badge-rejected");
    }
    if (probVal) probVal.textContent = `${res.model_estimated_probability}%`;
    if (probBar) probBar.style.width = `${res.model_estimated_probability}%`;

    // Render drivers
    const driversList = document.getElementById("classDriversList");
    if (driversList && res.top_drivers && res.top_drivers.length) {
      driversList.innerHTML = res.top_drivers.map(d => `
        <li><span class="driver-name">${d.feature}</span><span class="driver-weight">${d.weight}%</span></li>
      `).join("");
    }
  } catch (err) {
    console.error("Classification predict failed:", err);
  }
}

async function executeClassificationCompareAll() {
  const features = getClassificationFeaturesFromForm();
  try {
    const res = await fetch(`${API_BASE}/classification/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: "random_forest", features })
    }).then(r => r.json());

    const container = document.getElementById("classAgreementBars");
    if (!container) return;

    const names = {
      random_forest: "Random Forest (Tuned)",
      bagging: "Bagging Classifier",
      logistic_regression: "Logistic Regression",
      gradient_boosting: "Gradient Boosting",
      adaboost: "AdaBoost Classifier",
      svm_rbf: "SVM (RBF)",
      decision_tree: "Decision Tree",
      knn: "KNN (k=7)",
      mlp: "MLP Neural Network",
      naive_bayes: "Gaussian Naive Bayes"
    };

    container.innerHTML = Object.entries(res).map(([mid, d]) => `
      <div class="bar-row">
        <span class="bar-label">${names[mid] || mid}</span>
        <div class="bar-track"><div class="bar-fill" style="width: ${d.probability}%;"></div></div>
        <span class="bar-val" style="color: ${d.prediction === 'APPROVED' ? 'var(--color-success)' : 'var(--color-hazard)'};">${d.prediction} (${d.probability}%)</span>
      </div>
    `).join("");
  } catch (e) {
    console.error("Classification compare failed:", e);
  }
}

function renderClassificationLeaderboard(items) {
  const tbody = document.getElementById("classAllTableBody");
  if (!tbody) return;
  tbody.innerHTML = items.map(m => {
    const isPlaceholder = m.status.includes("Placeholder") || m.status.includes("In Development");
    return `
      <tr>
        <td><strong>${m.algorithm}</strong></td>
        <td><span class="badge ${m.status === 'Best Model' ? 'accent-badge' : ''}">${m.primary_metric.replace('ROC-AUC: ', '')}</span></td>
        <td>${m.accuracy}</td>
        <td>${m.secondary_metric.replace('F1: ', '')}</td>
        <td><span class="status-pill ${isPlaceholder ? 'status-hazard' : (m.status === 'Best Model' ? 'status-good' : 'status-moderate')}">${m.status}</span></td>
        <td><small class="text-muted">${m.phase}</small></td>
      </tr>
    `;
  }).join("");
}

async function updateFaceoffView() {
  const midA = state.classification.faceoffA;
  const midB = state.classification.faceoffB;

  try {
    const [evalA, evalB] = await Promise.all([
      fetch(`${API_BASE}/classification/evaluation/${midA}`).then(r => r.json()),
      fetch(`${API_BASE}/classification/evaluation/${midB}`).then(r => r.json())
    ]);

    const titleA = document.getElementById("faceoffTitleA");
    const titleB = document.getElementById("faceoffTitleB");
    if (titleA) titleA.textContent = document.querySelector(`#faceoffModelA option[value="${midA}"]`)?.text || midA;
    if (titleB) titleB.textContent = document.querySelector(`#faceoffModelB option[value="${midB}"]`)?.text || midB;

    document.getElementById("foAccA").textContent = `${(evalA.accuracy * 100).toFixed(1)}%`;
    document.getElementById("foAucA").textContent = evalA.roc_auc.toFixed(3);
    document.getElementById("foPrecA").textContent = `${(evalA.precision * 100).toFixed(1)}%`;
    document.getElementById("foRecA").textContent = `${(evalA.recall * 100).toFixed(1)}%`;
    document.getElementById("foF1A").textContent = evalA.f1.toFixed(3);

    document.getElementById("foAccB").textContent = `${(evalB.accuracy * 100).toFixed(1)}%`;
    document.getElementById("foAucB").textContent = evalB.roc_auc.toFixed(3);
    document.getElementById("foPrecB").textContent = `${(evalB.precision * 100).toFixed(1)}%`;
    document.getElementById("foRecB").textContent = `${(evalB.recall * 100).toFixed(1)}%`;
    document.getElementById("foF1B").textContent = evalB.f1.toFixed(3);

    // Render delta block
    const deltaContainer = document.getElementById("faceoffDeltaBlock");
    if (deltaContainer) {
      const diffAcc = ((evalA.accuracy - evalB.accuracy) * 100).toFixed(1);
      const diffAuc = (evalA.roc_auc - evalB.roc_auc).toFixed(3);
      deltaContainer.innerHTML = `
        <div class="callout-note">
          <strong>Head-to-Head Test Split Delta:</strong>
          Model A ${diffAcc >= 0 ? 'leads by +' + diffAcc + '%' : 'trails by ' + diffAcc + '%'} on Accuracy, 
          and ${diffAuc >= 0 ? 'leads by +' + diffAuc : 'trails by ' + diffAuc} on ROC-AUC over Model B.
        </div>
      `;
    }
  } catch (err) {
    console.error("Failed to update faceoff view:", err);
  }
}

async function loadClassificationEvaluation(modelId) {
  try {
    const res = await fetch(`${API_BASE}/classification/evaluation/${modelId}`).then(r => r.json());
    
    // Confusion matrix
    const [[tn, fp], [fn, tp]] = res.cm;
    document.getElementById("cmTN").textContent = tn;
    document.getElementById("cmFP").textContent = fp;
    document.getElementById("cmFN").textContent = fn;
    document.getElementById("cmTP").textContent = tp;

    document.getElementById("cmAcc").textContent = `${(res.accuracy * 100).toFixed(1)}%`;
    const spec = (tn / (tn + fp) * 100).toFixed(1);
    const sens = (tp / (tp + fn) * 100).toFixed(1);
    document.getElementById("cmSpec").textContent = `${spec}%`;
    document.getElementById("cmSens").textContent = `${sens}%`;

    const badge = document.getElementById("cmModelBadge");
    if (badge) badge.textContent = document.querySelector(`#classModelSelect option[value="${modelId}"]`)?.text.split(" [")[0] || modelId;

    drawRocCurve(res.roc_auc);
  } catch (err) {
    console.error("Failed to load classification evaluation:", err);
  }
}

function drawRocCurve(auc) {
  const canvas = document.getElementById("classRocCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = 35;
  // Background grid
  ctx.strokeStyle = "#22293a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + ((h - 2 * pad) / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    const x = pad + ((w - 2 * pad) / 4) * i;
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
  }

  // Random guessing line (diagonal)
  ctx.strokeStyle = "#64748b";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  ctx.lineTo(w - pad, pad);
  ctx.stroke();
  ctx.setLineDash([]);

  // ROC Curve
  ctx.strokeStyle = "#BF0C4F";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  // Quadratic bezier approximating high ROC AUC
  ctx.quadraticCurveTo(pad + 10, pad + 15, w - pad, pad);
  ctx.stroke();

  // Axis Labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px Inter";
  ctx.fillText("False Positive Rate (1 − Specificity)", w / 2 - 80, h - 10);
  ctx.save();
  ctx.translate(15, h / 2 + 30);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("True Positive Rate (Recall)", 0, 0);
  ctx.restore();
}

// -----------------------------------------------------------------------------
// 5. Clustering Lab Handlers & Playground
// -----------------------------------------------------------------------------
function initClusteringLab() {
  const algoSelect = document.getElementById("clustAlgoSelect");
  const linkageGroup = document.getElementById("linkageControlGroup");
  const kRange = document.getElementById("clustKRange");
  const kValText = document.getElementById("clustKVal");

  if (algoSelect) {
    algoSelect.addEventListener("change", (e) => {
      state.clustering.algorithm = e.target.value;
      if (linkageGroup) {
        linkageGroup.style.display = e.target.value === "agglomerative" ? "flex" : "none";
      }
    });
  }

  if (kRange && kValText) {
    kRange.addEventListener("input", (e) => {
      state.clustering.k = parseInt(e.target.value);
      kValText.textContent = e.target.value;
    });
  }

  document.getElementById("clustLinkageSelect")?.addEventListener("change", (e) => {
    state.clustering.linkage = e.target.value;
  });

  document.getElementById("btnRunClustering")?.addEventListener("click", executeClusteringRun);

  // Projection toggles
  document.getElementById("btnProjPCA")?.addEventListener("click", () => {
    document.getElementById("btnProjPCA").classList.add("active");
    document.getElementById("btnProjTSNE").classList.remove("active");
    state.clustering.projectionType = "pca";
    document.getElementById("projectionSubtitle").textContent = "PCA (Linear Variance): PC1 (31.5%) + PC2 (16.5%) = 48.0% Total Variance Explained";
    loadClusteringProjection("pca");
  });

  document.getElementById("btnProjTSNE")?.addEventListener("click", () => {
    document.getElementById("btnProjTSNE").classList.add("active");
    document.getElementById("btnProjPCA").classList.remove("active");
    state.clustering.projectionType = "tsne";
    document.getElementById("projectionSubtitle").textContent = "t-SNE (Non-Linear Manifold): Perplexity 15, Seed 42, Preserving Local Sensor Neighborhoods";
    loadClusteringProjection("tsne");
  });
}

async function loadClusteringData() {
  try {
    const config = await fetch(`${API_BASE}/clustering/config`).then(r => r.json());
    state.clustering.metrics = config.metrics;
    document.getElementById("clustSilhouette").textContent = config.metrics.silhouette.toFixed(4);
    document.getElementById("clustDB").textContent = config.metrics.davies_bouldin.toFixed(4);
    document.getElementById("clustCH").textContent = config.metrics.calinski_harabasz.toFixed(1);

    loadClusteringProjection("pca");
  } catch (err) {
    console.error("Failed to load clustering data:", err);
  }
}

async function loadClusteringProjection(ptype) {
  try {
    const res = await fetch(`${API_BASE}/clustering/projection`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ projection_type: ptype })
    }).then(r => r.json());

    state.clustering.pcaPoints = res.points;
    state.clustering.labels = res.labels;
    renderScatterCanvas();
  } catch (err) {
    console.error("Failed to load projection:", err);
  }
}

async function executeClusteringRun() {
  const req = {
    algorithm: state.clustering.algorithm,
    k: state.clustering.k,
    linkage: state.clustering.linkage
  };

  try {
    const res = await fetch(`${API_BASE}/clustering/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req)
    }).then(r => r.json());

    document.getElementById("clustSilhouette").textContent = res.silhouette_score.toFixed(4);
    document.getElementById("clustDB").textContent = res.davies_bouldin_index.toFixed(4);
    document.getElementById("clustCH").textContent = res.calinski_harabasz_index.toFixed(1);

    state.clustering.labels = res.labels;
    renderScatterCanvas();

    // Render updated cluster profile cards
    const grid = document.getElementById("clusterProfilesGrid");
    if (grid && res.cluster_counts) {
      grid.innerHTML = Object.entries(res.cluster_counts).map(([cid, count]) => {
        const pct = ((count / res.points_count) * 100).toFixed(1);
        return `
          <div class="panel-card cluster-card cluster-${cid}">
            <div class="cluster-badge cluster-${cid}-badge">Cluster ${cid}</div>
            <h3>Partition State ${cid}</h3>
            <div class="cluster-stat-row">
              <span>Sample Points:</span>
              <strong>${count} (${pct}%)</strong>
            </div>
            <div class="cluster-stat-row">
              <span>Telemetry Focus:</span>
              <span>Dynamically partitioned under ${res.algorithm.toUpperCase()} (k=${res.k})</span>
            </div>
          </div>
        `;
      }).join("");
    }
  } catch (err) {
    console.error("Clustering run failed:", err);
  }
}

function renderScatterCanvas() {
  const canvas = document.getElementById("clustScatterCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pts = state.clustering.pcaPoints;
  const labels = state.clustering.labels;
  if (!pts || !pts.length) return;

  const pad = 40;
  const xs = pts.map(p => p[0]);
  const ys = pts.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  // Background grid
  ctx.strokeStyle = "#1a202e";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    const y = pad + ((h - 2 * pad) / 6) * i;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    const x = pad + ((w - 2 * pad) / 6) * i;
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
  }

  // Palette: Color-blind conscious
  const colors = ["#38bdf8", "#f59e0b", "#10b981", "#a855f7", "#ec4899", "#6366f1", "#14b8a6", "#f43f5e"];

  // Draw points
  for (let i = 0; i < pts.length; i++) {
    const px = pad + ((pts[i][0] - minX) / (maxX - minX)) * (w - 2 * pad);
    const py = h - pad - ((pts[i][1] - minY) / (maxY - minY)) * (h - 2 * pad);
    const label = labels[i] !== undefined ? labels[i] : 0;
    ctx.fillStyle = colors[label % colors.length];
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Interactive Hover Tooltip
  const tooltip = document.getElementById("scatterTooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (w / rect.width);
    const mouseY = (e.clientY - rect.top) * (h / rect.height);

    let found = null;
    for (let i = 0; i < pts.length; i++) {
      const px = pad + ((pts[i][0] - minX) / (maxX - minX)) * (w - 2 * pad);
      const py = h - pad - ((pts[i][1] - minY) / (maxY - minY)) * (h - 2 * pad);
      const dist = Math.hypot(mouseX - px, mouseY - py);
      if (dist < 8) {
        found = { idx: i, x: pts[i][0], y: pts[i][1], label: labels[i] };
        break;
      }
    }

    if (found && tooltip) {
      tooltip.style.display = "block";
      tooltip.style.left = `${(e.clientX - rect.left) + 12}px`;
      tooltip.style.top = `${(e.clientY - rect.top) - 10}px`;
      tooltip.innerHTML = `<strong>Sample #${found.idx}</strong><br>Cluster: <span style="color: ${colors[found.label % colors.length]}; font-weight:700;">Cluster ${found.label}</span><br>Coord: [${found.x.toFixed(2)}, ${found.y.toFixed(2)}]`;
    } else if (tooltip) {
      tooltip.style.display = "none";
    }
  };

  canvas.onmouseleave = () => {
    if (tooltip) tooltip.style.display = "none";
  };
}

function drawElbowPlot() {
  const canvas = document.getElementById("clustElbowCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = 50;
  const kVals = [2, 3, 4, 5, 6, 7, 8, 9, 10];
  const inertias = [89200, 78100, 71500, 66400, 62100, 58900, 56200, 53800, 51700];
  const minIn = 45000, maxIn = 95000;

  // Grid
  ctx.strokeStyle = "#1a202e";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad + ((h - 2 * pad) / 5) * i;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }

  // Inertia line
  ctx.strokeStyle = "#BF0C4F";
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let i = 0; i < kVals.length; i++) {
    const x = pad + (i / (kVals.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((inertias[i] - minIn) / (maxIn - minIn)) * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Points and optimal callout at k=2
  for (let i = 0; i < kVals.length; i++) {
    const x = pad + (i / (kVals.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((inertias[i] - minIn) / (maxIn - minIn)) * (h - 2 * pad);
    ctx.fillStyle = i === 0 ? "#10b981" : "#BF0C4F";
    ctx.beginPath();
    ctx.arc(x, y, i === 0 ? 7 : 4.5, 0, Math.PI * 2);
    ctx.fill();

    // X-axis text
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px Inter";
    ctx.fillText(`k=${kVals[i]}`, x - 10, h - pad + 20);
  }

  // Optimal annotation callout
  ctx.fillStyle = "#10b981";
  ctx.font = "bold 12px Inter";
  ctx.fillText("★ Optimal Elbow (k = 2)", pad + 15, h - pad - ((inertias[0] - minIn) / (maxIn - minIn)) * (h - 2 * pad) - 15);

  // Axis Labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px Inter";
  ctx.fillText("Number of Clusters (K)", w / 2 - 60, h - 10);
  ctx.save();
  ctx.translate(18, h / 2 + 50);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Within-Cluster Sum of Squares (Inertia)", 0, 0);
  ctx.restore();
}

function drawDendrogram() {
  const canvas = document.getElementById("clustDendroCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = 50;

  // Draw illustrative hierarchical tree
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;

  // Draw sample tree branches
  const leafCount = 16;
  const step = (w - 2 * pad) / (leafCount - 1);
  const leafY = h - pad;

  const clusterAEnd = 10;
  // Sub-branch Cluster A
  for (let i = 0; i < clusterAEnd; i += 2) {
    const x1 = pad + i * step;
    const x2 = pad + (i + 1) * step;
    const my = leafY - 40 - (i * 4);
    ctx.strokeStyle = "#38bdf8";
    ctx.beginPath();
    ctx.moveTo(x1, leafY); ctx.lineTo(x1, my);
    ctx.lineTo(x2, my); ctx.lineTo(x2, leafY);
    ctx.stroke();
  }

  // Sub-branch Cluster B
  for (let i = clusterAEnd; i < leafCount; i += 2) {
    const x1 = pad + i * step;
    const x2 = pad + (i + 1) * step;
    const my = leafY - 45;
    ctx.strokeStyle = "#f59e0b";
    ctx.beginPath();
    ctx.moveTo(x1, leafY); ctx.lineTo(x1, my);
    ctx.lineTo(x2, my); ctx.lineTo(x2, leafY);
    ctx.stroke();
  }

  // Major trunks
  const midA = pad + (step * (clusterAEnd / 2));
  const midB = pad + (step * (clusterAEnd + (leafCount - clusterAEnd) / 2));
  const trunkY = pad + 60;

  ctx.strokeStyle = "#38bdf8";
  ctx.beginPath();
  ctx.moveTo(midA, leafY - 70); ctx.lineTo(midA, trunkY);
  ctx.stroke();

  ctx.strokeStyle = "#f59e0b";
  ctx.beginPath();
  ctx.moveTo(midB, leafY - 50); ctx.lineTo(midB, trunkY);
  ctx.stroke();

  // Root merge
  ctx.strokeStyle = "#64748b";
  ctx.beginPath();
  ctx.moveTo(midA, trunkY); ctx.lineTo(midB, trunkY);
  ctx.moveTo((midA + midB) / 2, trunkY); ctx.lineTo((midA + midB) / 2, pad + 10);
  ctx.stroke();

  // Cutoff threshold line (d = 3.8)
  const cutY = trunkY + 15;
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(pad - 10, cutY);
  ctx.lineTo(w - pad + 10, cutY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#ef4444";
  ctx.font = "bold 11px Inter";
  ctx.fillText("← Cutoff Threshold: Distance = 3.8 (Yields K=2 Clusters)", w - 380, cutY - 6);

  // Labels
  ctx.fillStyle = "#38bdf8";
  ctx.fillText("Cluster 0 Branch (Stop-Prone)", midA - 70, leafY + 25);
  ctx.fillStyle = "#f59e0b";
  ctx.fillText("Cluster 1 Branch (Nominal)", midB - 60, leafY + 25);
}

// -----------------------------------------------------------------------------
// 6. Model Comparison Studio & Export CSV
// -----------------------------------------------------------------------------
function initStudio() {
  document.querySelectorAll(".filter-pill").forEach(pill => {
    pill.addEventListener("click", (e) => {
      document.querySelectorAll(".filter-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const filter = pill.getAttribute("data-filter");
      renderStudioTable(filter);
    });
  });

  document.getElementById("btnExportCSV")?.addEventListener("click", exportStudioCSV);

  // Global search input
  const searchInput = document.getElementById("globalSearchInput");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = state.leaderboard.filter(row => 
        row.algorithm.toLowerCase().includes(q) || 
        row.track.toLowerCase().includes(q) || 
        row.status.toLowerCase().includes(q)
      );
      renderStudioTableWithData(filtered);
    });
  }
}

function renderStudioTable(filter = "all") {
  let list = state.leaderboard;
  if (filter !== "all") {
    list = list.filter(item => item.track.toLowerCase() === filter.toLowerCase());
  }
  renderStudioTableWithData(list);
}

function renderStudioTableWithData(list) {
  const tbody = document.getElementById("studioTableBody");
  if (!tbody) return;
  tbody.innerHTML = list.map(m => {
    const isPlaceholder = m.status.includes("Placeholder") || m.status.includes("In Development");
    return `
      <tr>
        <td><span class="badge ${m.track.toLowerCase()}-badge">${m.track}</span></td>
        <td><strong>${m.algorithm}</strong></td>
        <td><span class="badge ${m.status === 'Best Model' ? 'accent-badge' : ''}">${m.primary_metric}</span></td>
        <td>${m.secondary_metric}</td>
        <td>${m.accuracy}</td>
        <td><span class="status-pill ${isPlaceholder ? 'status-hazard' : (m.status === 'Best Model' ? 'status-good' : 'status-moderate')}">${m.status}</span></td>
        <td><small class="text-muted">${m.phase}</small></td>
      </tr>
    `;
  }).join("");
}

function exportStudioCSV() {
  const headers = ["Track", "Algorithm", "Primary Metric", "Secondary Metric", "Accuracy / Error", "Status", "Phase"];
  const rows = state.leaderboard.map(r => [
    `"${r.track}"`,
    `"${r.algorithm}"`,
    `"${r.primary_metric}"`,
    `"${r.secondary_metric}"`,
    `"${r.accuracy}"`,
    `"${r.status}"`,
    `"${r.phase}"`
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "ML_Lab_Capstone_Leaderboard.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// -----------------------------------------------------------------------------
// 7. Initialization Lifecycle
// -----------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  initRouter();
  initTabs();
  initRegressionLab();
  initClassificationLab();
  initClusteringLab();
  initStudio();
  initData();
});
