const API_BASE = "/api";
const AUTH = "Bearer admin-token";
const DEFAULT_CORPUS = "manuals_vendor_keyence";
const STORAGE_KEY = "manuals-rag-last-eval-result";
const ASSET_VERSION = "20260821-latest-completed-1";

const state = {
  documents: [],
  latestRun: null,
  activeRun: null,
  currentEval: null,
  selectedEvalIndex: 0,
  running: false,
  runDebug: null,
  runDebugTimer: null,
  evalRuntime: null,
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

async function localJson(path) {
  const response = await fetch(path, { headers: { "Cache-Control": "no-cache" } });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return response.json();
}

function setStatus(message, mode = "idle") {
  const node = $("eval-status");
  node.textContent = message;
  node.className = `status-pill ${mode}`;
}

function renderMetrics(summary = {}) {
  const total = Number(summary.total_questions ?? 0);
  const retrievalCorrect = Number(summary.retrieval_correct ?? 0);
  const answersCorrect = Number(summary.answers_correct ?? 0);
  $("eval-summary").className = "metrics";
  $("eval-summary").innerHTML = [
    ["Rows", total],
    ["Retrieval", `${retrievalCorrect}/${total} (${Number(summary.retrieval_correct_percent ?? 0).toFixed(2)}%)`],
    ["Answers", `${answersCorrect}/${total} (${Number(summary.answers_correct_percent ?? 0).toFixed(2)}%)`],
    ["Failures", Math.max(0, total - answersCorrect)],
  ]
    .map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function summarizeVisibleItems(items = []) {
  const total = items.length;
  const retrievalCorrect = items.filter((item) => item.retrieval_evaluation?.passed).length;
  const answersCorrect = items.filter((item) => item.answer_evaluation?.passed).length;
  return {
    total_questions: total,
    retrieval_correct: retrievalCorrect,
    retrieval_correct_percent: total ? (retrievalCorrect / total) * 100 : 0,
    answers_correct: answersCorrect,
    answers_correct_percent: total ? (answersCorrect / total) * 100 : 0,
  };
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
                <td data-label="#">${index + 1}</td>
                <td data-label="Question">${escapeHtml(item.case?.query)}</td>
                <td data-label="Retrieval"><span class="badge ${retrieval.passed ? "pass" : "fail"}">${retrieval.passed ? "pass" : "fail"}</span></td>
                <td data-label="Rank">${escapeHtml(retrieval.rank ?? "not found")}</td>
                <td data-label="Answer"><span class="badge ${answerEval.passed ? "pass" : "fail"}">${answerEval.passed ? "pass" : "fail"}</span></td>
                <td data-label="Failures">${escapeHtml((answerEval.failure_reasons || []).join(", "))}</td>
                <td data-label="Answer Preview">${escapeHtml(shortText(item.answer?.answer, 180))}</td>
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
                <td data-label="Document">${escapeHtml(citation.document_id)}</td>
                <td data-label="Chunk">${escapeHtml(citation.chunk_id)}</td>
                <td data-label="Pages">${escapeHtml((citation.pages || []).join(", "))}</td>
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
                <td data-label="Rank">${index + 1}</td>
                <td data-label="Score">${Number(result.score ?? 0).toFixed(4)}</td>
                <td data-label="Title">${escapeHtml(result.title)}</td>
                <td data-label="Pages">${escapeHtml((result.pages || []).join(", "))}</td>
                <td data-label="Preview">${escapeHtml(shortText(result.content || result.content_preview, 260))}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function getQuestionTrace(runtime, questionIndex) {
  const key = String(questionIndex || "unknown");
  runtime.questions[key] ||= {
    index: questionIndex,
    total: null,
    case: null,
    status: "pending",
    llmCalls: {},
    retrieved: [],
    answer: null,
  };
  return runtime.questions[key];
}

function summarizeLlmCall(call = {}) {
  const text = String(call.text || "").trim();
  if (!text) return "";
  try {
    const parsed = JSON.parse(text);
    if (parsed.answer) return parsed.answer;
    if (parsed.summary) return parsed.summary;
    return JSON.stringify(parsed, null, 2);
  } catch {
    return text;
  }
}

function renderQuestionTrace(runtime) {
  const node = $("question-trace");
  if (!runtime) {
    $("trace-status").textContent = "No active trace.";
    node.className = "question-trace empty-state";
    node.textContent = "Run or resume an eval to inspect each question.";
    return;
  }
  const traces = Object.values(runtime.questions).sort((a, b) => Number(a.index || 0) - Number(b.index || 0));
  $("trace-status").textContent = traces.length ? `${traces.length} questions seen` : "Waiting for questions.";
  if (!traces.length) {
    node.className = "question-trace empty-state";
    node.textContent = "Waiting for the first eval question.";
    return;
  }
  node.className = "question-trace";
  node.innerHTML = traces
    .map((trace, traceIndex) => {
      const caseData = trace.case || {};
      const llmCalls = Object.entries(trace.llmCalls || {});
      const open = trace.status === "running" || traceIndex === traces.length - 1;
      return `
        <details class="trace-card" ${open ? "open" : ""}>
          <summary>
            <span>Question ${escapeHtml(trace.index)}${trace.total ? `/${escapeHtml(trace.total)}` : ""}</span>
            <span class="badge ${trace.status === "completed" ? "pass" : ""}">${escapeHtml(trace.status)}</span>
          </summary>
          <div class="trace-grid">
            <section>
              <h4>Source Chunk</h4>
              <dl>
                <dt>Document</dt><dd>${escapeHtml(caseData.source_filename || "")}</dd>
                <dt>Pages</dt><dd>${escapeHtml([caseData.page_from, caseData.page_to].filter(Boolean).join("-"))}</dd>
                <dt>Chunk</dt><dd>${escapeHtml(caseData.source_chunk_id || "")}</dd>
                <dt>Terms</dt><dd>${escapeHtml((caseData.expected_terms || []).join(", "))}</dd>
              </dl>
              <pre>${escapeHtml(caseData.expected_snippet || "")}</pre>
            </section>
            <section>
              <h4>Generated Question</h4>
              <p class="question">${escapeHtml(caseData.query || "")}</p>
              <h4>Generated Answer</h4>
              <p class="answer-text">${escapeHtml(trace.answer?.answer || "")}</p>
            </section>
          </div>
          <h4>Retrieved Content</h4>
          ${renderTopResults((trace.retrieved || []).slice(0, 5))}
          <h4>Model Streams</h4>
          <div class="trace-streams">
            ${
              llmCalls.length
                ? llmCalls
                    .map(([callId, call]) => `
                      <article class="model-call">
                        <div class="model-meta">${escapeHtml(call.label || callId)} | ${escapeHtml(call.model || "")} | ${escapeHtml(call.status || "running")}</div>
                        <pre>${escapeHtml(summarizeLlmCall(call))}</pre>
                      </article>
                    `)
                    .join("")
                : '<div class="empty-state">No model generation has started for this question yet.</div>'
            }
          </div>
        </details>
      `;
    })
    .join("");
}

function scheduleQuestionTraceRender(runtime) {
  if (runtime.traceRenderTimer) return;
  runtime.traceRenderTimer = setTimeout(() => {
    runtime.traceRenderTimer = null;
    renderQuestionTrace(runtime);
  }, 250);
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
  const items = payload.items || [];
  payload = { ...payload, summary: summarizeVisibleItems(items) };
  state.currentEval = payload;
  if (state.selectedEvalIndex >= items.length) state.selectedEvalIndex = 0;
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ payload, meta, savedAt: new Date().toISOString() }));
  $("result-run-id").textContent = meta.id ? `Run ${meta.id}` : payload.run_id ? `Run ${payload.run_id}` : "";
  renderMetrics(payload.summary || {});
  renderEvalTable(payload);
  renderEvalDetail(payload);
}

function runtimeFromEvalResult(payload, runId = null) {
  const runtime = createEvalRuntime();
  runtime.runId = runId || payload?.run_id || null;
  for (const [index, item] of (payload?.items || []).entries()) {
    const trace = getQuestionTrace(runtime, index + 1);
    trace.total = payload.items.length;
    trace.case = item.case || null;
    trace.status = "completed";
    trace.retrieved = item.top_results || [];
    trace.answer = item.answer || null;
  }
  return runtime;
}

function renderCompletedEvalRun(run, source) {
  if (!run?.result_json) return;
  state.selectedEvalIndex = 0;
  renderEval(run.result_json, { id: run.id, updated_at: run.updated_at, source });
  const runtime = runtimeFromEvalResult(run.result_json, run.id);
  state.evalRuntime = runtime;
  $("model-output").innerHTML = "";
  renderQuestionTrace(runtime);
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

function renderEvalWarnings(warnings = []) {
  if (!warnings.length) return;
  $("progress-list").innerHTML = warnings
    .map((warning) => `<div class="error-box">${escapeHtml(warning)}</div>`)
    .join("");
}

function resetRunDebug(payload, sampleLimit) {
  state.runDebug = {
    startedAt: Date.now(),
    requestUrl: `${API_BASE}/eval/end-to-end-stream?sample_limit=${sampleLimit}`,
    payload,
    sampleLimit,
    runId: null,
    httpStatus: null,
    bytes: 0,
    chunks: 0,
    events: 0,
    lastEvent: "not connected",
    lastEventAt: null,
    lines: [],
  };
  $("run-debug-log").textContent = "";
  appendRunDebug("Prepared eval request", { requestUrl: state.runDebug.requestUrl, payload });
  renderRunDebug();
}

function appendRunDebug(message, details = null) {
  if (!state.runDebug) return;
  const elapsed = ((Date.now() - state.runDebug.startedAt) / 1000).toFixed(1);
  const suffix = details ? ` ${JSON.stringify(details)}` : "";
  state.runDebug.lines.push(`[+${elapsed}s] ${message}${suffix}`);
  state.runDebug.lines = state.runDebug.lines.slice(-80);
  $("run-debug-log").textContent = state.runDebug.lines.join("\n");
  $("run-debug-log").scrollTop = $("run-debug-log").scrollHeight;
}

function updateRunDebug(patch = {}) {
  if (!state.runDebug) return;
  Object.assign(state.runDebug, patch);
  renderRunDebug();
}

function renderRunDebug() {
  const debug = state.runDebug;
  if (!debug) {
    $("run-debug-summary").className = "debug-summary empty-state";
    $("run-debug-summary").textContent = "No active run.";
    return;
  }
  const elapsed = ((Date.now() - debug.startedAt) / 1000).toFixed(1);
  const lastAge = debug.lastEventAt ? `${((Date.now() - debug.lastEventAt) / 1000).toFixed(1)}s ago` : "never";
  $("run-debug-summary").className = "debug-summary metrics compact";
  $("run-debug-summary").innerHTML = [
    ["Run", debug.runId || "pending"],
    ["HTTP", debug.httpStatus || "opening"],
    ["Events", debug.events],
    ["Bytes", debug.bytes],
    ["Chunks", debug.chunks],
    ["Last Event", debug.lastEvent],
    ["Last Seen", lastAge],
    ["Elapsed", `${elapsed}s`],
  ]
    .map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function summarizeEvent(event) {
  const queryEvent = event.query_event || {};
  return {
    event: event.event,
    run_id: event.run_id,
    question_index: event.question_index,
    total_questions: event.total_questions,
    nested_event: queryEvent.event,
    step: queryEvent.step,
    call_id: queryEvent.call_id,
    label: queryEvent.label,
    model: queryEvent.model,
    error: event.error || queryEvent.error,
  };
}

function createEvalRuntime() {
  return {
    llmOutputs: {},
    questions: {},
    traceRenderTimer: null,
    stepSequence: [],
    stepState: {},
    finalResult: null,
    runId: null,
    lastEventIndex: 0,
  };
}

function processEvalEvent(event, runtime, source = "stream", eventIndex = null) {
  runtime.runId = event.run_id || runtime.runId;
  if (eventIndex !== null) {
    runtime.lastEventIndex = Math.max(runtime.lastEventIndex, Number(eventIndex) || 0);
  }
  if (state.runDebug) {
    updateRunDebug({
      runId: runtime.runId,
      events: state.runDebug.events + 1,
      lastEvent: event.query_event?.event ? `${event.event}:${event.query_event.event}` : event.event,
      lastEventAt: Date.now(),
    });
    if (event.query_event?.event !== "llm_token") {
      appendRunDebug(`Event from ${source}`, summarizeEvent(event));
    }
  }
  handleEvalEvent(event, {
    runtime,
    llmOutputs: runtime.llmOutputs,
    stepSequenceRef: (value) => (runtime.stepSequence = value),
    stepStateRef: (value) => (runtime.stepState = value),
  });
  if (event.event === "eval_completed") runtime.finalResult = event.result;
}

async function pollRunToCompletion(runtime) {
  if (!runtime.runId) return;
  appendRunDebug("Polling persisted run after stream interruption", { runId: runtime.runId, after: runtime.lastEventIndex });
  setStatus(`Reconnected to run ${runtime.runId}`, "running");
  const deadline = Date.now() + 20 * 60 * 1000;
  while (Date.now() < deadline) {
    const rows = await localJson(`/local/run-events?run_id=${encodeURIComponent(runtime.runId)}&after=${runtime.lastEventIndex}&limit=1000`);
    appendRunDebug("Polled run events", { count: rows.length, after: runtime.lastEventIndex });
    for (const row of rows) {
      processEvalEvent(row.event_json, runtime, "poll", row.event_index);
    }
    const run = await apiJson(`/runs/${runtime.runId}`);
    if (run.status === "completed" && run.result_json) {
      appendRunDebug("Persisted run completed", { status: run.status, eventsSeen: runtime.lastEventIndex });
      runtime.finalResult = run.result_json;
      setStatus("Completed", "complete");
      renderCompletedEvalRun(run, "persisted run after reconnect");
      await loadLatestRun();
      return;
    }
    if (run.status === "failed") {
      throw new Error(run.error || "Persisted run failed");
    }
    if (rows.length >= 1000) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      continue;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error(`Timed out waiting for persisted run ${runtime.runId}`);
}

async function resumeEvalRun(runId) {
  if (!runId || state.running) return;
  state.running = true;
  $("run-eval").disabled = true;
  $("model-output").innerHTML = "";
  $("progress-list").innerHTML = "";
  renderQuestionTrace(null);
  state.selectedEvalIndex = 0;
  resetRunDebug({ resumed_run_id: runId }, Number($("sample-limit").value || 10));
  appendRunDebug("Resuming persisted run", { runId });
  if (state.runDebugTimer) clearInterval(state.runDebugTimer);
  state.runDebugTimer = setInterval(renderRunDebug, 1000);
  const runtime = createEvalRuntime();
  runtime.runId = runId;
  state.evalRuntime = runtime;
  try {
    const run = await apiJson(`/runs/${runId}`);
    updateRunDebug({ runId, httpStatus: "polling persisted run" });
    if (run.result_json) {
      renderCompletedEvalRun(run, "history");
      setStatus(run.status === "completed" ? "Loaded completed run" : `Loaded ${run.status} run`, run.status === "failed" ? "error" : "complete");
      return;
    }
    if (run.progress_json?.event) {
      processEvalEvent(run.progress_json, runtime, "progress");
    }
    await pollRunToCompletion(runtime);
  } catch (error) {
    appendRunDebug("Resume failed", { message: error.message, name: error.name });
    setStatus("Failed", "error");
    $("progress-list").innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  } finally {
    if (state.runDebugTimer) {
      clearInterval(state.runDebugTimer);
      state.runDebugTimer = null;
    }
    renderRunDebug();
    state.running = false;
    $("run-eval").disabled = false;
  }
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

function updateQuestionTrace(event, runtime) {
  let immediate = true;
  if (event.event === "eval_started") {
    runtime.questions = {};
    renderQuestionTrace(runtime);
    return;
  }
  if (!event.question_index) return;
  const trace = getQuestionTrace(runtime, event.question_index);
  if (event.total_questions) trace.total = event.total_questions;

  if (event.event === "eval_question_started") {
    trace.case = event.case || trace.case;
    trace.status = "running";
  } else if (event.event === "eval_query_event") {
    const queryEvent = event.query_event || {};
    if (queryEvent.event === "llm_call_started") {
      trace.llmCalls[queryEvent.call_id] = {
        label: queryEvent.label,
        model: queryEvent.model,
        status: "running",
        purpose: queryEvent.purpose,
        text: "",
      };
    } else if (queryEvent.event === "llm_token") {
      const call = (trace.llmCalls[queryEvent.call_id] ||= { status: "running", text: "" });
      call.text = `${call.text || ""}${queryEvent.token || ""}`;
      immediate = false;
    } else if (queryEvent.event === "llm_call_completed") {
      const call = (trace.llmCalls[queryEvent.call_id] ||= { text: "" });
      call.label = queryEvent.label || call.label;
      call.model = queryEvent.model || call.model;
      call.status = "completed";
      call.text = queryEvent.raw_response || call.text || "";
    } else if (queryEvent.event === "llm_call_failed") {
      const call = (trace.llmCalls[queryEvent.call_id] ||= { text: "" });
      call.status = "failed";
      call.text = queryEvent.error || call.text || "";
    } else if (queryEvent.event === "step_completed" && queryEvent.payload?.answer) {
      trace.answer = queryEvent.payload.answer;
    } else if (queryEvent.event === "run_completed" && queryEvent.result) {
      trace.retrieved = queryEvent.result.stages?.find((stage) => stage.name === "retrieval_results")?.samples || trace.retrieved;
      trace.answer = queryEvent.result.answer || trace.answer;
    }
  } else if (event.event === "eval_question_completed") {
    trace.status = "completed";
    trace.case = event.item?.case || trace.case;
    trace.answer = event.item?.answer || trace.answer;
    trace.retrieved = event.item?.top_results || trace.retrieved;
  } else if (event.event === "eval_failed") {
    trace.status = "failed";
  }
  if (immediate) {
    renderQuestionTrace(runtime);
  } else {
    scheduleQuestionTraceRender(runtime);
  }
}

async function runEval() {
  if (state.running) return;
  state.running = true;
  $("run-eval").disabled = true;
  $("model-output").innerHTML = "";
  $("progress-list").innerHTML = "";
  renderQuestionTrace(null);
  state.currentEval = null;
  $("eval-summary").className = "metrics empty-state";
  $("eval-summary").textContent = "No evaluation loaded.";
  $("eval-table").innerHTML = "";
  $("eval-detail").innerHTML = "";
  setStatus("Starting", "running");
  const documentId = $("eval-document").value;
  let scope = $("eval-scope").value;
  if (documentId && scope !== "document") {
    scope = "document";
    $("eval-scope").value = "document";
  }
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
  resetRunDebug(payload, sampleLimit);
  if (state.runDebugTimer) clearInterval(state.runDebugTimer);
  state.runDebugTimer = setInterval(renderRunDebug, 1000);
  const runtime = createEvalRuntime();
  state.evalRuntime = runtime;
  try {
    appendRunDebug("Opening stream");
    const response = await apiFetch(`/eval/end-to-end-stream?sample_limit=${sampleLimit}`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    updateRunDebug({ httpStatus: `${response.status} ${response.statusText}` });
    appendRunDebug("Stream opened", {
      status: response.status,
      contentType: response.headers.get("content-type"),
      cacheControl: response.headers.get("cache-control"),
    });
    if (!response.body) {
      throw new Error("Response body is not readable; this browser/WebView may not support streaming fetch bodies.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        appendRunDebug("Stream reader ended", { bufferedCharacters: buffer.length });
        break;
      }
      updateRunDebug({
        bytes: state.runDebug.bytes + value.length,
        chunks: state.runDebug.chunks + 1,
      });
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      appendRunDebug("Received stream chunk", { bytes: value.length, completeLines: lines.filter((line) => line.trim()).length });
      for (const line of lines) {
        if (!line.trim()) continue;
        let event;
        try {
          event = JSON.parse(line);
        } catch (error) {
          appendRunDebug("Could not parse stream line", { message: error.message, line: line.slice(0, 300) });
          throw error;
        }
        processEvalEvent(event, runtime, "stream");
      }
    }
    if (runtime.finalResult) {
      appendRunDebug("Rendering final streamed result");
      setStatus("Completed", "complete");
      renderEval(runtime.finalResult, { id: runtime.runId, source: "current streamed run" });
      await loadLatestRun();
    } else if (runtime.runId) {
      appendRunDebug("No final result event; fetching persisted run", { runId: runtime.runId });
      const run = await apiJson(`/runs/${runtime.runId}`);
      appendRunDebug("Persisted run loaded", { status: run.status, hasResult: Boolean(run.result_json) });
      if (run.status === "completed" && run.result_json) {
        setStatus("Completed", "complete");
        renderCompletedEvalRun(run, "persisted run");
      } else if (run.status === "running" || run.status === "queued") {
        await pollRunToCompletion(runtime);
      } else {
        setStatus(run.status || "Ended without final result", run.status === "failed" ? "error" : "idle");
      }
    } else {
      appendRunDebug("Stream ended before a run id was received");
    }
  } catch (error) {
    appendRunDebug("Run failed", { message: error.message, name: error.name });
    if (runtime.runId) {
      try {
        await pollRunToCompletion(runtime);
      } catch (pollError) {
        appendRunDebug("Polling failed", { message: pollError.message, name: pollError.name });
        setStatus("Failed", "error");
        $("progress-list").innerHTML = `<div class="error-box">${escapeHtml(pollError.message)}</div>`;
      }
    } else {
      setStatus("Failed", "error");
      $("progress-list").innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
    }
  } finally {
    appendRunDebug("Run controls released");
    if (state.runDebugTimer) {
      clearInterval(state.runDebugTimer);
      state.runDebugTimer = null;
    }
    renderRunDebug();
    state.running = false;
    $("run-eval").disabled = false;
  }
}

function handleEvalEvent(event, refs) {
  if (event.event === "eval_started") {
    state.currentEval = { summary: summarizeVisibleItems([]), items: [], warnings: event.warnings || [] };
    renderMetrics(state.currentEval.summary);
    $("eval-table").innerHTML = '<div class="empty-state">Waiting for completed questions.</div>';
    $("eval-detail").innerHTML = "";
    setStatus(`Run ${event.run_id}: ${event.total_questions} questions`, "running");
    renderEvalWarnings(event.warnings || []);
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
  if (refs.runtime) updateQuestionTrace(event, refs.runtime);
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
  const runs = await apiJson("/runs?run_type=end_to_end_eval&limit=25&include_result=false");
  state.activeRun = runs.find((run) => run.status === "running" || run.status === "queued") || null;
  state.latestRun = runs.find((run) => run.status === "completed" && Number(run.progress_json?.summary?.total_questions || 0) > 0) || null;
  if (!state.latestRun) {
    $("latest-run").innerHTML = state.activeRun
      ? `No completed eval output yet.<br><span class="muted">Active run ${escapeHtml(state.activeRun.id)} is ${escapeHtml(state.activeRun.status)}.</span>`
      : "No completed eval output yet.";
    return;
  }
  const progress = state.latestRun.progress_json || {};
  const summary = progress.summary || {};
  const items = Number(summary.total_questions || 0);
  const activeHtml = state.activeRun
    ? `<div class="muted">Active: ${escapeHtml(state.activeRun.id)} (${escapeHtml(state.activeRun.status)})</div>`
    : "";
  $("latest-run").innerHTML = `
    <div><strong>${escapeHtml(state.latestRun.id)}</strong></div>
    <div>Latest completed output</div>
    <div>Updated ${escapeHtml(state.latestRun.updated_at)}</div>
    <div>${items} rows | ${Number(summary.retrieval_correct_percent || 0).toFixed(2)}% retrieval | ${Number(summary.answers_correct_percent || 0).toFixed(2)}% answers</div>
    ${activeHtml}
  `;
}

async function loadLatestResults() {
  if (!state.latestRun?.id) return;
  const run = await apiJson(`/runs/${state.latestRun.id}`);
  if (!run.result_json) return;
  state.latestRun = run;
  renderCompletedEvalRun(run, "latest persisted completed run");
  setStatus("Loaded latest completed run", "complete");
}

function runSummaryText(run) {
  const summary = run.progress_json?.summary || {};
  const total = Number(summary.total_questions || 0);
  if (!total) return "0 rows";
  return `${total} rows | R ${Number(summary.retrieval_correct_percent || 0).toFixed(0)}% | A ${Number(summary.answers_correct_percent || 0).toFixed(0)}%`;
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
  const runs = await apiJson("/runs?limit=50&include_result=false");
  $("history-table").innerHTML = `
    <table>
      <thead><tr><th>Updated</th><th>Type</th><th>Status</th><th>Summary</th><th>Run ID</th><th>Error</th></tr></thead>
      <tbody>
        ${runs
          .map(
            (run) => `
              <tr class="clickable" data-run-id="${run.id}">
                <td data-label="Updated">${escapeHtml(run.updated_at)}</td>
                <td data-label="Type">${escapeHtml(run.run_type)}</td>
                <td data-label="Status">${escapeHtml(run.status)}</td>
                <td data-label="Summary">${escapeHtml(runSummaryText(run))}</td>
                <td data-label="Run ID">${escapeHtml(run.id)}</td>
                <td data-label="Error">${escapeHtml(shortText(run.error, 120))}</td>
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
      if (run.run_type === "end_to_end_eval" && (run.result_json || run.status === "running" || run.status === "queued")) {
        state.selectedEvalIndex = 0;
        document.querySelector('[data-tab="eval"]').click();
        if (run.result_json) {
          renderCompletedEvalRun(run, "history");
          setStatus(`Loaded ${run.status} run`, run.status === "failed" ? "error" : "complete");
        } else {
          await resumeEvalRun(run.id);
        }
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

function setupEvalScopeControls() {
  $("eval-document").addEventListener("change", () => {
    if ($("eval-document").value) {
      $("eval-scope").value = "document";
    }
  });
  $("eval-scope").addEventListener("change", () => {
    if ($("eval-scope").value === "corpus") {
      $("eval-document").value = "";
    }
  });
}

async function init() {
  setupTabs();
  setupEvalScopeControls();
  $("run-eval").addEventListener("click", runEval);
  $("load-latest").addEventListener("click", loadLatestResults);
  $("run-query").addEventListener("click", runQuery);
  $("refresh-history").addEventListener("click", loadHistory);
  try {
    await loadDocuments();
    await loadLatestRun();
    await loadHistory();
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const { payload, meta } = JSON.parse(saved);
      renderEval(payload, meta);
    }
    $("connection-status").textContent = `API connected at ${API_BASE}`;
  } catch (error) {
    $("connection-status").textContent = `API error: ${error.message}`;
    $("connection-status").className = "error-text";
  }
}

document.querySelector('link[href^="/styles.css"]').href = `/styles.css?v=${ASSET_VERSION}`;

init();
