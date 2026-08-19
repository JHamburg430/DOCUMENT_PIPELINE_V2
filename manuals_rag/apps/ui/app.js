const API_BASE = "/api";
const AUTH = "Bearer admin-token";
const DEFAULT_CORPUS = "manuals_vendor_keyence";
const STORAGE_KEY = "manuals-rag-last-eval-result";

const state = {
  documents: [],
  latestRun: null,
  currentEval: null,
  selectedEvalIndex: 0,
  running: false,
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function shortText(value, limit = 220) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1).trim()}...` : text;
}

function splitList(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      Authorization: AUTH,
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return response;
}

async function apiJson(path, options = {}) {
  const response = await apiFetch(path, options);
  return response.json();
}

function setStatus(message, mode = "idle") {
  const node = $("eval-status");
  node.textContent = message;
  node.className = `status-pill ${mode}`;
}

function renderMetrics(summary = {}) {
  $("eval-summary").className = "metrics";
  $("eval-summary").innerHTML = [
    ["Questions", summary.total_questions ?? 0],
    ["Retrieval", `${Number(summary.retrieval_correct_percent ?? 0).toFixed(2)}%`],
    ["Answers", `${Number(summary.answers_correct_percent ?? 0).toFixed(2)}%`],
    ["Answer Passes", summary.answers_correct ?? 0],
  ]
    .map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function renderEvalTable(payload) {
  const items = payload?.items || [];
  if (!items.length) {
    $("eval-table").innerHTML = '<div class="empty-state">No evaluation items returned.</div>';
    $("eval-detail").innerHTML = "";
    return;
  }
  $("eval-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Question</th>
          <th>Retrieval</th>
          <th>Rank</th>
          <th>Answer</th>
          <th>Failures</th>
          <th>Answer Preview</th>
        </tr>
      </thead>
      <tbody>
        ${items
          .map((item, index) => {
            const retrieval = item.retrieval_evaluation || {};
            const answerEval = item.answer_evaluation || {};
            const selected = index === state.selectedEvalIndex ? " selected" : "";
            return `
              <tr class="clickable${selected}" data-eval-index="${index}">
                <td>${index + 1}</td>
                <td>${escapeHtml(item.case?.query)}</td>
                <td><span class="badge ${retrieval.passed ? "pass" : "fail"}">${retrieval.passed ? "pass" : "fail"}</span></td>
                <td>${escapeHtml(retrieval.rank ?? "not found")}</td>
                <td><span class="badge ${answerEval.passed ? "pass" : "fail"}">${answerEval.passed ? "pass" : "fail"}</span></td>
                <td>${escapeHtml((answerEval.failure_reasons || []).join(", "))}</td>
                <td>${escapeHtml(shortText(item.answer?.answer, 180))}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
  document.querySelectorAll("[data-eval-index]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedEvalIndex = Number(row.dataset.evalIndex);
      renderEval(state.currentEval);
    });
  });
}

function renderList(items) {
  if (!items?.length) return '<span class="muted">none</span>';
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function renderCitations(citations = []) {
  if (!citations.length) return '<div class="muted">No citations returned.</div>';
  return `
    <table>
      <thead><tr><th>Document</th><th>Chunk</th><th>Pages</th></tr></thead>
      <tbody>
        ${citations
          .map(
            (citation) => `
              <tr>
                <td>${escapeHtml(citation.document_id)}</td>
                <td>${escapeHtml(citation.chunk_id)}</td>
                <td>${escapeHtml((citation.pages || []).join(", "))}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderTopResults(results = []) {
  if (!results.length) return '<div class="empty-state">No top search results recorded.</div>';
  return `
    <table>
      <thead><tr><th>Rank</th><th>Score</th><th>Title</th><th>Pages</th><th>Preview</th></tr></thead>
      <tbody>
        ${results
          .map(
            (result, index) => `
              <tr>
                <td>${index + 1}</td>
                <td>${Number(result.score ?? 0).toFixed(4)}</td>
                <td>${escapeHtml(result.title)}</td>
                <td>${escapeHtml((result.pages || []).join(", "))}</td>
                <td>${escapeHtml(shortText(result.content || result.content_preview, 260))}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderEvalDetail(payload) {
  const item = payload?.items?.[state.selectedEvalIndex];
  if (!item) {
    $("eval-detail").innerHTML = "";
    return;
  }
  const caseData = item.case || {};
  const answer = item.answer || {};
  const retrieval = item.retrieval_evaluation || {};
  const answerEval = item.answer_evaluation || {};
  const termCheck = answerEval.term_check || {};
  $("eval-detail").innerHTML = `
    <section class="detail">
      <h3>Question ${state.selectedEvalIndex + 1}</h3>
      <p class="question">${escapeHtml(caseData.query)}</p>
      <div class="metrics compact">
        <div class="metric"><span>Retrieval</span><strong>${retrieval.passed ? "pass" : "fail"}</strong></div>
        <div class="metric"><span>Rank</span><strong>${escapeHtml(retrieval.rank ?? "not found")}</strong></div>
        <div class="metric"><span>Answer</span><strong>${answerEval.passed ? "pass" : "fail"}</strong></div>
        <div class="metric"><span>Expected Terms</span><strong>${termCheck.passed ? "pass" : "fail"}</strong></div>
      </div>
      <div class="split">
        <div>
          <h4>Expected</h4>
          <dl>
            <dt>Document</dt><dd>${escapeHtml(caseData.source_filename)}</dd>
            <dt>Title</dt><dd>${escapeHtml(caseData.source_title)}</dd>
            <dt>Pages</dt><dd>${escapeHtml([caseData.page_from, caseData.page_to].filter(Boolean).join("-"))}</dd>
            <dt>Terms</dt><dd>${escapeHtml((caseData.expected_terms || []).join(", "))}</dd>
          </dl>
          <h4>Expected Snippet</h4>
          <pre>${escapeHtml(caseData.expected_snippet)}</pre>
        </div>
        <div>
          <h4>Generated Answer</h4>
          <p class="answer-text">${escapeHtml(answer.answer || "")}</p>
          <h4>Citations</h4>
          ${renderCitations(answer.citations || [])}
          <h4>Failure Reasons</h4>
          ${renderList([...(retrieval.failure_reasons || []), ...(answerEval.failure_reasons || [])])}
        </div>
      </div>
      <h4>Top Search Results</h4>
      ${renderTopResults(item.top_results || [])}
      <details>
        <summary>Raw selected item JSON</summary>
        <pre>${escapeHtml(JSON.stringify(item, null, 2))}</pre>
      </details>
    </section>
  `;
}

function renderEval(payload, meta = {}) {
  if (!payload) return;
  state.currentEval = payload;
  if (state.selectedEvalIndex >= (payload.items || []).length) state.selectedEvalIndex = 0;
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ payload, meta, savedAt: new Date().toISOString() }));
  $("result-run-id").textContent = meta.id ? `Run ${meta.id}` : payload.run_id ? `Run ${payload.run_id}` : "";
  renderMetrics(payload.summary || {});
  renderEvalTable(payload);
  renderEvalDetail(payload);
}

function renderProgress(stepSequence = [], stepState = {}) {
  if (!stepSequence.length) {
    $("progress-list").innerHTML = '<div class="empty-state">No active question progress.</div>';
    return;
  }
  $("progress-list").innerHTML = stepSequence
    .map((step, index) => {
      const status = stepState[step.name]?.status || "pending";
      return `<div class="progress-row ${status}"><span>${index + 1}. ${escapeHtml(step.label)}</span><strong>${escapeHtml(status)}</strong></div>`;
    })
    .join("");
}

function appendModelOutput(callId, call) {
  const node = $("model-output");
  let block = node.querySelector(`[data-call-id="${CSS.escape(callId)}"]`);
  if (!block) {
    block = document.createElement("div");
    block.className = "model-call";
    block.dataset.callId = callId;
    node.appendChild(block);
  }
  let body = `<div class="model-meta">${escapeHtml(call.label || callId)} | ${escapeHtml(call.model || "")} | ${escapeHtml(call.status || "running")}</div>`;
  const text = String(call.text || "");
  try {
    const parsed = JSON.parse(text);
    body += parsed.answer ? `<p>${escapeHtml(parsed.answer)}</p>` : "";
    if (parsed.citations?.length) body += renderCitations(parsed.citations);
    body += `<details><summary>Raw model payload</summary><pre>${escapeHtml(JSON.stringify(parsed, null, 2))}</pre></details>`;
  } catch {
    body += `<pre>${escapeHtml(text)}</pre>`;
  }
  block.innerHTML = body;
  node.scrollTop = node.scrollHeight;
}

async function runEval() {
  if (state.running) return;
  state.running = true;
  $("run-eval").disabled = true;
  $("model-output").innerHTML = "";
  $("progress-list").innerHTML = "";
  setStatus("Starting", "running");
  const scope = $("eval-scope").value;
  const documentId = $("eval-document").value;
  const payload = {
    corpus_ids: splitList($("eval-corpus").value),
    document_id: scope === "document" ? documentId : null,
    max_questions: Number($("eval-max-questions").value || 8),
    use_llm_generation: $("eval-llm-generation").checked,
  };
  if (scope === "document" && !documentId) {
    setStatus("Choose a document", "error");
    state.running = false;
    $("run-eval").disabled = false;
    return;
  }
  const sampleLimit = Number($("sample-limit").value || 10);
  const llmOutputs = {};
  let stepSequence = [];
  let stepState = {};
  let finalResult = null;
  let runId = null;
  try {
    const response = await apiFetch(`/eval/end-to-end-stream?sample_limit=${sampleLimit}`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        runId = event.run_id || runId;
        handleEvalEvent(event, { llmOutputs, stepSequenceRef: (value) => (stepSequence = value), stepStateRef: (value) => (stepState = value) });
        if (event.event === "eval_completed") finalResult = event.result;
      }
    }
    if (finalResult) {
      setStatus("Completed", "complete");
      renderEval(finalResult, { id: runId, source: "current streamed run" });
      await loadLatestRun();
    } else if (runId) {
      const run = await apiJson(`/runs/${runId}`);
      if (run.status === "completed" && run.result_json) {
        setStatus("Completed", "complete");
        renderEval(run.result_json, { id: run.id, updated_at: run.updated_at, source: "persisted run" });
      } else {
        setStatus(run.status || "Ended without final result", run.status === "failed" ? "error" : "idle");
      }
    }
  } catch (error) {
    setStatus("Failed", "error");
    $("progress-list").innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  } finally {
    state.running = false;
    $("run-eval").disabled = false;
  }
}

function handleEvalEvent(event, refs) {
  if (event.event === "eval_started") {
    setStatus(`Run ${event.run_id}: ${event.total_questions} questions`, "running");
  } else if (event.event === "eval_question_started") {
    setStatus(`Question ${event.question_index}/${event.total_questions}`, "running");
  } else if (event.event === "eval_query_event") {
    const queryEvent = event.query_event || {};
    if (queryEvent.event === "run_started") {
      refs.stepSequenceRef(queryEvent.step_sequence || []);
      refs.stepStateRef({});
      renderProgress(queryEvent.step_sequence || [], {});
    } else if (queryEvent.event === "step_started") {
      const current = Object.fromEntries([...document.querySelectorAll(".progress-row")].map((row) => [row.textContent, row.className]));
      void current;
    }
    updateNestedProgress(queryEvent);
    updateModelOutput(event.question_index, queryEvent, refs.llmOutputs);
  } else if (event.event === "eval_question_completed") {
    renderEval({ summary: event.summary, items: [...(state.currentEval?.items || []), event.item], warnings: [] }, { id: event.run_id, source: "partial run" });
  } else if (event.event === "eval_failed") {
    setStatus(event.error || "Failed", "error");
  }
}

const progressState = { sequence: [], state: {} };

function updateNestedProgress(queryEvent) {
  if (queryEvent.event === "run_started") {
    progressState.sequence = queryEvent.step_sequence || [];
    progressState.state = {};
  } else if (queryEvent.event === "step_started") {
    progressState.state[queryEvent.step] = { status: "running" };
  } else if (queryEvent.event === "step_completed") {
    progressState.state[queryEvent.step] = { status: "completed" };
  }
  renderProgress(progressState.sequence, progressState.state);
}

function updateModelOutput(questionIndex, queryEvent, llmOutputs) {
  const nested = queryEvent.event;
  const baseId = `q${questionIndex}:${queryEvent.call_id}`;
  if (nested === "llm_call_started") {
    llmOutputs[baseId] = { label: queryEvent.label, model: queryEvent.model, status: "running", text: "" };
    appendModelOutput(baseId, llmOutputs[baseId]);
  } else if (nested === "llm_token") {
    const call = (llmOutputs[baseId] ||= { status: "running", text: "" });
    call.text = `${call.text || ""}${queryEvent.token || ""}`;
    appendModelOutput(baseId, call);
  } else if (nested === "llm_call_completed") {
    const call = (llmOutputs[baseId] ||= { text: "" });
    call.label = queryEvent.label || call.label;
    call.model = queryEvent.model || call.model;
    call.status = "completed";
    call.text = queryEvent.raw_response || call.text || "";
    appendModelOutput(baseId, call);
  }
}

async function loadDocuments() {
  state.documents = await apiJson("/debug/documents?limit=200");
  const select = $("eval-document");
  select.innerHTML = '<option value="">Choose document...</option>';
  for (const doc of state.documents) {
    const option = document.createElement("option");
    option.value = doc.document_id;
    option.textContent = `${doc.title || doc.source_filename} | ${doc.source_filename}`;
    select.appendChild(option);
  }
}

async function loadLatestRun() {
  const runs = await apiJson("/runs?run_type=end_to_end_eval&limit=25");
  state.latestRun =
    runs.find((run) => run.status === "completed" && run.result_json?.items?.length) ||
    runs.find((run) => run.status === "completed" && run.result_json) ||
    null;
  if (!state.latestRun) {
    $("latest-run").innerHTML = "No completed runs yet.";
    return;
  }
  const summary = state.latestRun.result_json.summary || {};
  $("latest-run").innerHTML = `
    <div><strong>${escapeHtml(state.latestRun.id)}</strong></div>
    <div>Updated ${escapeHtml(state.latestRun.updated_at)}</div>
    <div>${Number(summary.retrieval_correct_percent || 0).toFixed(2)}% retrieval | ${Number(summary.answers_correct_percent || 0).toFixed(2)}% answers</div>
  `;
}

function loadLatestResults() {
  if (!state.latestRun?.result_json) return;
  state.selectedEvalIndex = 0;
  renderEval(state.latestRun.result_json, {
    id: state.latestRun.id,
    updated_at: state.latestRun.updated_at,
    source: "latest persisted completed run",
  });
  setStatus("Loaded latest completed run", "complete");
}

async function runQuery() {
  const node = $("answer");
  node.textContent = "Running...";
  try {
    const payload = await apiJson("/query", {
      method: "POST",
      body: JSON.stringify({
        query: $("query").value.trim(),
        corpus_ids: splitList($("query-corpus").value || DEFAULT_CORPUS),
        filters: {},
        response_mode: "answer_with_citations",
      }),
    });
    node.innerHTML = `
      <h3>Answer</h3>
      <p>${escapeHtml(payload.answer || "")}</p>
      <h3>Citations</h3>
      ${renderCitations(payload.citations || [])}
      <details><summary>Raw answer JSON</summary><pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre></details>
    `;
  } catch (error) {
    node.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
}

async function loadHistory() {
  const runs = await apiJson("/runs?limit=50");
  $("history-table").innerHTML = `
    <table>
      <thead><tr><th>Updated</th><th>Type</th><th>Status</th><th>Run ID</th><th>Error</th></tr></thead>
      <tbody>
        ${runs
          .map(
            (run) => `
              <tr class="clickable" data-run-id="${run.id}">
                <td>${escapeHtml(run.updated_at)}</td>
                <td>${escapeHtml(run.run_type)}</td>
                <td>${escapeHtml(run.status)}</td>
                <td>${escapeHtml(run.id)}</td>
                <td>${escapeHtml(shortText(run.error, 120))}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
  document.querySelectorAll("[data-run-id]").forEach((row) => {
    row.addEventListener("click", async () => {
      const run = await apiJson(`/runs/${row.dataset.runId}`);
      $("history-detail").innerHTML = `<details open><summary>Run JSON</summary><pre>${escapeHtml(JSON.stringify(run, null, 2))}</pre></details>`;
      if (run.run_type === "end_to_end_eval" && run.result_json) {
        state.selectedEvalIndex = 0;
        renderEval(run.result_json, { id: run.id, updated_at: run.updated_at, source: "history" });
        document.querySelector('[data-tab="eval"]').click();
      }
    });
  });
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      $(tab.dataset.tab).classList.add("active");
    });
  });
}

async function init() {
  setupTabs();
  $("run-eval").addEventListener("click", runEval);
  $("load-latest").addEventListener("click", loadLatestResults);
  $("run-query").addEventListener("click", runQuery);
  $("refresh-history").addEventListener("click", loadHistory);
  try {
    await loadDocuments();
    await loadLatestRun();
    await loadHistory();
    if (state.latestRun?.result_json) {
      loadLatestResults();
    } else {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const { payload, meta } = JSON.parse(saved);
        renderEval(payload, meta);
      }
    }
    $("connection-status").textContent = `API connected at ${API_BASE}`;
  } catch (error) {
    $("connection-status").textContent = `API error: ${error.message}`;
    $("connection-status").className = "error-text";
  }
}

init();
