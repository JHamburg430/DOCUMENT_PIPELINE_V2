const API_BASE = window.location.origin.replace(":8601", ":8600");
const AUTH = "Bearer admin-token";

async function runQuery() {
  const query = document.getElementById("query").value.trim();
  const answerNode = document.getElementById("answer");
  answerNode.textContent = "Running...";
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: AUTH,
    },
    body: JSON.stringify({
      query,
      corpus_ids: ["manuals_vendor_keyence"],
      filters: {},
      response_mode: "answer_with_citations",
    }),
  });
  answerNode.textContent = JSON.stringify(await response.json(), null, 2);
}

async function uploadFiles() {
  const files = document.getElementById("file").files;
  const node = document.getElementById("upload-result");
  node.textContent = "Uploading...";
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const response = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    headers: { Authorization: AUTH },
    body: form,
  });
  const uploaded = await response.json();
  node.textContent = JSON.stringify(uploaded, null, 2);
}

document.getElementById("run-query").addEventListener("click", runQuery);
document.getElementById("upload").addEventListener("click", uploadFiles);
