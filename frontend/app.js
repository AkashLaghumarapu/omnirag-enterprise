/**
 * OmniRAG Enterprise - Client-Side Controller
 * Manages Server-Sent Events (SSE) streaming, telemetry meters,
 * citation inspector, and interactive enterprise benchmarking.
 */

// Application State
const state = {
  activeSources: [],
  currentQuery: '',
  isStreaming: false,
  selectedDocId: null,
  selectedDocTitle: 'Searching All Documents',
  config: {
    topK: 5,
    enableCRAG: true,
    enableRerank: true,
    provider: 'auto',
    apiKey: ''
  }
};

// DOM Elements
const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const btnSend = document.getElementById('btn-send');
const statChunks = document.getElementById('stat-chunks');
const statEngine = document.getElementById('stat-engine');
const docList = document.getElementById('doc-list');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const sliderTopK = document.getElementById('slider-topk');
const valTopK = document.getElementById('val-topk');
const toggleCRAG = document.getElementById('toggle-crag');
const toggleRerank = document.getElementById('toggle-rerank');

// Observability Elements
const cragBadge = document.getElementById('crag-badge');
const cragExplanation = document.getElementById('crag-explanation');
const queryRewritesContainer = document.getElementById('query-rewrites');
const rewriteList = document.getElementById('rewrite-list');
const totalLatencyBadge = document.getElementById('total-latency');

const barTransform = document.getElementById('bar-transform');
const valTransform = document.getElementById('val-transform');
const barHybrid = document.getElementById('bar-hybrid');
const valHybrid = document.getElementById('val-hybrid');
const barRRF = document.getElementById('bar-rrf');
const valRRF = document.getElementById('val-rrf');
const barRerank = document.getElementById('bar-rerank');
const valRerank = document.getElementById('val-rerank');
const barCRAG = document.getElementById('bar-crag');
const valCRAG = document.getElementById('val-crag');
const barLLM = document.getElementById('bar-llm');
const valLLM = document.getElementById('val-llm');

const triadStatus = document.getElementById('triad-status');
const circlePrecision = document.getElementById('circle-precision');
const scorePrecision = document.getElementById('score-precision');
const circleFaithfulness = document.getElementById('circle-faithfulness');
const scoreFaithfulness = document.getElementById('score-faithfulness');
const circleRelevance = document.getElementById('circle-relevance');
const scoreRelevance = document.getElementById('score-relevance');

const sourcesContainer = document.getElementById('sources-container');
const sourcesCount = document.getElementById('sources-count');

// Modals
const provenanceModal = document.getElementById('provenance-modal');
const archModal = document.getElementById('architecture-modal');
const settingsModal = document.getElementById('settings-modal');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  // Restore settings from localStorage if available
  const savedGemini = localStorage.getItem('omnirag_gemini_key');
  const savedGroq = localStorage.getItem('omnirag_groq_key');
  const savedProvider = localStorage.getItem('omnirag_provider');
  if (savedGemini) {
    const el = document.getElementById('gemini-key');
    if (el) el.value = savedGemini;
    state.config.apiKey = savedGemini;
  }
  if (savedGroq) {
    const el = document.getElementById('groq-key');
    if (el) el.value = savedGroq;
  }
  if (savedProvider) {
    const el = document.getElementById('provider-select');
    if (el) el.value = savedProvider;
    state.config.provider = savedProvider;
  }

  // Sync settings with backend
  if (savedGemini || savedGroq || savedProvider) {
    fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        gemini_key: savedGemini || null,
        groq_key: savedGroq || null,
        provider: savedProvider || 'auto'
      })
    }).then(() => fetchSystemStatus()).catch(console.error);
  }

  initEventListeners();
  fetchSystemStatus();
  fetchDocumentList();
});

function initEventListeners() {
  // Chat submit
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleQuerySubmit();
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit();
    }
  });

  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(120, Math.max(44, chatInput.scrollHeight)) + 'px';
  });

  // Prompt chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      chatInput.value = chip.dataset.query;
      handleQuerySubmit();
    });
  });

  // Hyperparameters
  sliderTopK.addEventListener('input', (e) => {
    state.config.topK = parseInt(e.target.value);
    valTopK.textContent = state.config.topK;
  });

  toggleCRAG.addEventListener('change', (e) => {
    state.config.enableCRAG = e.target.checked;
  });

  toggleRerank.addEventListener('change', (e) => {
    state.config.enableRerank = e.target.checked;
  });

  // Sample Loaders
  document.getElementById('load-sample-nvidia').addEventListener('click', () => loadSample('sec_10k_nvidia'));
  document.getElementById('load-sample-rag').addEventListener('click', () => loadSample('rag_architecture_spec'));

  // Clear Corpus & Samples
  document.getElementById('btn-clear-samples').addEventListener('click', clearSamples);
  document.getElementById('btn-clear-corpus').addEventListener('click', clearCorpus);
  document.getElementById('btn-scope-all').addEventListener('click', () => setDocumentScope(null, 'Searching All Documents'));

  // File Upload
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileUpload);
  
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-active'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    if (e.dataTransfer.files.length) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  // Modals
  document.getElementById('btn-architecture-modal').addEventListener('click', () => archModal.classList.remove('hidden'));
  document.getElementById('btn-close-arch-modal').addEventListener('click', () => archModal.classList.add('hidden'));

  document.getElementById('btn-settings-modal').addEventListener('click', () => settingsModal.classList.remove('hidden'));
  document.getElementById('btn-close-settings').addEventListener('click', () => settingsModal.classList.add('hidden'));

  document.getElementById('btn-close-modal').addEventListener('click', () => provenanceModal.classList.add('hidden'));

  // Architecture Tabs
  document.querySelectorAll('.arch-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.arch-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.arch-tab-content').forEach(c => c.classList.add('hidden'));
      tab.classList.add('active');
      document.getElementById(`tab-${tab.dataset.tab}`).classList.remove('hidden');
    });
  });

  // Settings Save
  document.getElementById('btn-save-settings').addEventListener('click', saveSettings);
}

// ----------------- API Interactions -----------------

async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    statChunks.textContent = data.vector_store.total_chunks;
    
    if (data.active_models.gemini_configured) {
      statEngine.textContent = 'Gemini 1.5 Flash';
      statEngine.className = 'metric-val text-cyan';
    } else if (data.active_models.groq_configured) {
      statEngine.textContent = 'Groq Llama-3.3';
      statEngine.className = 'metric-val text-purple';
    } else {
      statEngine.textContent = 'Offline Semantic Mode';
      statEngine.className = 'metric-val text-emerald';
    }
  } catch (err) {
    console.error('Failed to fetch status:', err);
  }
}

async function fetchDocumentList() {
  try {
    const res = await fetch('/api/documents/list');
    const data = await res.json();
    renderDocumentList(data.documents);
    statChunks.textContent = data.total_chunks;
  } catch (err) {
    console.error('Failed to fetch docs:', err);
  }
}

function setDocumentScope(docId, docTitle) {
  state.selectedDocId = docId;
  state.selectedDocTitle = docTitle || 'Searching All Documents';

  const scopeTarget = document.getElementById('scope-target-name');
  const btnScopeAll = document.getElementById('btn-scope-all');
  if (scopeTarget) scopeTarget.textContent = state.selectedDocTitle;
  if (btnScopeAll) {
    if (!docId) {
      btnScopeAll.classList.add('active-scope');
    } else {
      btnScopeAll.classList.remove('active-scope');
    }
  }

  // Update doc items active class
  document.querySelectorAll('.doc-badge-item').forEach(el => {
    if (docId && el.dataset.docId === docId) {
      el.classList.add('active-doc');
    } else {
      el.classList.remove('active-doc');
    }
  });

  appendSystemNotification(`Search focus shifted to: <strong>${escapeHTML(state.selectedDocTitle)}</strong>`);
}

function renderDocumentList(docs) {
  docList.innerHTML = '';
  if (!docs || docs.length === 0) {
    docList.innerHTML = '<div class="empty-sources">No documents indexed. Load a sample or upload above.</div>';
    setDocumentScope(null, 'No Documents Indexed');
    return;
  }

  // If currently selected docId is no longer in docs, reset to null
  if (state.selectedDocId && !docs.some(d => d.doc_id === state.selectedDocId)) {
    state.selectedDocId = null;
    state.selectedDocTitle = 'Searching All Documents';
    const scopeTarget = document.getElementById('scope-target-name');
    if (scopeTarget) scopeTarget.textContent = state.selectedDocTitle;
  }

  docs.forEach(doc => {
    const el = document.createElement('div');
    el.className = `doc-badge-item ${state.selectedDocId === doc.doc_id ? 'active-doc' : ''}`;
    el.dataset.docId = doc.doc_id;

    el.innerHTML = `
      <div class="doc-badge-top">
        <div class="doc-badge-title" title="${escapeHTML(doc.title)}">${escapeHTML(doc.title)}</div>
        <div class="doc-badge-actions">
          <button class="btn-delete-doc" title="Delete this document">&times;</button>
        </div>
      </div>
      <div class="doc-badge-sub">
        <span>${doc.chunk_count} chunks</span>
        <span class="text-cyan">${state.selectedDocId === doc.doc_id ? '● Active Target' : 'Click to Focus'}</span>
      </div>
    `;

    // Click to focus
    el.addEventListener('click', () => {
      setDocumentScope(doc.doc_id, doc.title);
    });

    // Click delete button
    const delBtn = el.querySelector('.btn-delete-doc');
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteDocument(doc.doc_id, doc.title);
    });

    docList.appendChild(el);
  });
}

async function clearSamples() {
  try {
    const res = await fetch('/api/documents/clear-samples', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      appendSystemNotification('Removed sample documents. Keeping your uploaded files.');
      await fetchSystemStatus();
      await fetchDocumentList();
    }
  } catch (err) {
    alert('Failed to clear samples: ' + err.message);
  }
}

async function deleteDocument(docId, docTitle) {
  if (!confirm(`Delete "${docTitle}" from the vector index?`)) return;
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      appendSystemNotification(`Deleted: <strong>${escapeHTML(docTitle)}</strong>`);
      if (state.selectedDocId === docId) {
        setDocumentScope(null, 'Searching All Documents');
      }
      await fetchSystemStatus();
      await fetchDocumentList();
    }
  } catch (err) {
    alert('Failed to delete document: ' + err.message);
  }
}

async function loadSample(sampleKey) {
  try {
    const res = await fetch('/api/documents/load-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_key: sampleKey })
    });
    const data = await res.json();
    if (data.success) {
      fetchSystemStatus();
      fetchDocumentList();
      appendSystemNotification(`Loaded sample: <strong>${data.message}</strong>`);
    }
  } catch (err) {
    alert('Failed to load sample: ' + err.message);
  }
}

async function handleFileUpload(e) {
  if (e.target.files.length) {
    await uploadFile(e.target.files[0]);
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  appendSystemNotification(`Uploading and indexing: <em>${escapeHTML(file.name)}</em>...`);

  try {
    const res = await fetch('/api/documents/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.success) {
      await fetchSystemStatus();
      await fetchDocumentList();
      if (data.doc_id) {
        setDocumentScope(data.doc_id, data.title);
      }
      appendSystemNotification(`Indexed <strong>${data.chunks_created}</strong> chunks from <em>${escapeHTML(data.filename)}</em>. Search scope is now focused on this document!`);
    }
  } catch (err) {
    alert('Failed to upload file: ' + err.message);
  }
}

async function clearCorpus() {
  if (!confirm('Are you sure you want to clear all indexed documents from the vector store?')) return;
  try {
    await fetch('/api/documents/clear', { method: 'POST' });
    fetchSystemStatus();
    fetchDocumentList();
    sourcesContainer.innerHTML = '<div class="empty-sources">Index cleared.</div>';
    sourcesCount.textContent = '0 Chunks';
    appendSystemNotification('Corpus index has been reset.');
  } catch (err) {
    alert('Error clearing corpus: ' + err.message);
  }
}

async function saveSettings() {
  const geminiKey = document.getElementById('gemini-key').value.trim();
  const groqKey = document.getElementById('groq-key').value.trim();
  const provider = document.getElementById('provider-select').value;

  // Persist locally in browser storage
  if (geminiKey) localStorage.setItem('omnirag_gemini_key', geminiKey);
  else localStorage.removeItem('omnirag_gemini_key');

  if (groqKey) localStorage.setItem('omnirag_groq_key', groqKey);
  else localStorage.removeItem('omnirag_groq_key');

  localStorage.setItem('omnirag_provider', provider);
  state.config.apiKey = geminiKey;
  state.config.provider = provider;

  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        gemini_key: geminiKey || null,
        groq_key: groqKey || null,
        provider: provider
      })
    });
    const data = await res.json();
    if (data.success) {
      settingsModal.classList.add('hidden');
      fetchSystemStatus();
      appendSystemNotification('API and provider configuration saved and active.');
    }
  } catch (err) {
    alert('Error saving settings: ' + err.message);
  }
}

// ----------------- Query Execution & Streaming -----------------

async function handleQuerySubmit() {
  const query = chatInput.value.trim();
  if (!query || state.isStreaming) return;

  state.currentQuery = query;
  state.isStreaming = true;
  chatInput.value = '';
  chatInput.style.height = '44px';
  btnSend.disabled = true;

  // Append user message
  appendUserMessage(query);

  // Append pending assistant message
  const assistantMsgObj = createAssistantMessageElement();
  chatHistory.appendChild(assistantMsgObj.container);
  chatHistory.scrollTop = chatHistory.scrollHeight;

  // Reset telemetry UI
  resetTelemetry();

  try {
    const activeApiKey = state.config.apiKey || localStorage.getItem('omnirag_gemini_key') || '';
    const activeProvider = state.config.provider || localStorage.getItem('omnirag_provider') || 'auto';

    const response = await fetch('/api/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        top_k: state.config.topK,
        enable_crag: state.config.enableCRAG,
        enable_reranking: state.config.enableRerank,
        provider: activeProvider,
        api_key: activeApiKey,
        doc_id: state.selectedDocId
      })
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || 'Query execution failed');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let accumulatedText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // keep last partial line

      for (const block of lines) {
        if (!block.trim()) continue;
        const eventMatch = block.match(/^event:\s*(.+)$/m);
        const dataMatch = block.match(/^data:\s*(.+)$/m);

        if (!eventMatch || !dataMatch) continue;

        const eventType = eventMatch[1].trim();
        const eventData = JSON.parse(dataMatch[1].trim());

        if (eventType === 'metadata') {
          handleMetadataEvent(eventData);
        } else if (eventType === 'token') {
          accumulatedText += eventData.token;
          renderStreamingMarkdown(assistantMsgObj.contentEl, accumulatedText);
          chatHistory.scrollTop = chatHistory.scrollHeight;
        } else if (eventType === 'benchmark') {
          handleBenchmarkEvent(eventData);
        } else if (eventType === 'done') {
          state.isStreaming = false;
        }
      }
    }

  } catch (err) {
    assistantMsgObj.contentEl.innerHTML = `<span class="text-red">Error: ${escapeHTML(err.message)}</span>`;
  } finally {
    state.isStreaming = false;
    btnSend.disabled = false;
  }
}

// ----------------- Event Handlers -----------------

function handleMetadataEvent(data) {
  // CRAG Status Card
  const crag = data.crag;
  cragBadge.textContent = crag.grade;
  cragBadge.className = `badge-status ${
    crag.grade === 'CORRECT' ? 'badge-pass' : crag.grade === 'AMBIGUOUS' ? 'badge-warn' : 'badge-fail'
  }`;
  cragExplanation.textContent = crag.explanation;

  // Query Rewrites
  if (data.expanded_queries && data.expanded_queries.length > 1) {
    queryRewritesContainer.classList.remove('hidden');
    rewriteList.innerHTML = '';
    data.expanded_queries.forEach(q => {
      const li = document.createElement('li');
      li.textContent = q;
      rewriteList.appendChild(li);
    });
  } else {
    queryRewritesContainer.classList.add('hidden');
  }

  // Render Sources in right panel
  state.activeSources = data.sources;
  sourcesCount.textContent = `${data.sources.length} Chunks`;
  sourcesContainer.innerHTML = '';

  if (data.sources.length === 0) {
    sourcesContainer.innerHTML = '<div class="empty-sources">No chunks met the relevance threshold.</div>';
    return;
  }

  data.sources.forEach(src => {
    const el = document.createElement('div');
    el.className = 'source-item';
    el.innerHTML = `
      <div class="source-item-header">
        <div class="source-item-title"><span class="citation-badge">[${src.citation_index}]</span> ${escapeHTML(src.doc_title)}</div>
        <div class="source-item-score">Rel: ${(src.rerank_score * 100).toFixed(0)}%</div>
      </div>
      <div class="source-item-snippet">${escapeHTML(src.content_preview)}</div>
    `;
    el.addEventListener('click', () => openProvenanceModal(src));
    sourcesContainer.appendChild(el);
  });
}

function handleBenchmarkEvent(benchmark) {
  // Latency Waterfall
  const lat = benchmark.latency_breakdown_ms;
  const total = benchmark.total_latency_ms;
  totalLatencyBadge.textContent = `${total.toFixed(0)} ms`;

  updateBar(barTransform, valTransform, lat.query_transformation_ms, total);
  updateBar(barHybrid, valHybrid, lat.hybrid_retrieval_ms, total);
  updateBar(barRRF, valRRF, lat.rrf_fusion_ms, total);
  updateBar(barRerank, valRerank, lat.cross_encoder_rerank_ms, total);
  updateBar(barCRAG, valCRAG, lat.crag_evaluation_ms, total);
  updateBar(barLLM, valLLM, lat.llm_generation_ms, total);

  // RAG Triad Gauges
  const prec = benchmark.context_precision.score * 100;
  const faith = benchmark.faithfulness.score * 100;
  const rel = benchmark.answer_relevance.score * 100;

  updateCircleMeter(circlePrecision, scorePrecision, prec);
  updateCircleMeter(circleFaithfulness, scoreFaithfulness, faith);
  updateCircleMeter(circleRelevance, scoreRelevance, rel);

  triadStatus.textContent = benchmark.status;
  triadStatus.className = `badge-status ${benchmark.status === 'PASS' ? 'badge-pass' : 'badge-warn'}`;
}

function updateBar(barEl, valEl, valMs, totalMs) {
  if (!barEl || !valEl) return;
  const ms = valMs || 0;
  const pct = totalMs > 0 ? Math.min(100, Math.max(2, (ms / totalMs) * 100)) : 0;
  barEl.style.width = `${pct}%`;
  valEl.textContent = `${ms.toFixed(1)}ms`;
}

function updateCircleMeter(circleEl, textEl, percentage) {
  const rounded = Math.round(percentage);
  circleEl.setAttribute('stroke-dasharray', `${rounded}, 100`);
  textEl.textContent = `${rounded}%`;
}

function resetTelemetry() {
  totalLatencyBadge.textContent = 'Measuring...';
  [valTransform, valHybrid, valRRF, valRerank, valCRAG, valLLM].forEach(el => el.textContent = '--');
  [barTransform, barHybrid, barRRF, barRerank, barCRAG, barLLM].forEach(el => el.style.width = '0%');
}

// ----------------- UI Rendering Helpers -----------------

function appendUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'chat-message user-message';
  msg.innerHTML = `
    <div class="message-body">
      <div class="message-content">${escapeHTML(text)}</div>
    </div>
  `;
  chatHistory.appendChild(msg);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function createAssistantMessageElement() {
  const container = document.createElement('div');
  container.className = 'chat-message bot-message';
  container.innerHTML = `
    <div class="message-avatar">
      <div class="avatar-ring"></div>
      <span>AI</span>
    </div>
    <div class="message-body">
      <div class="message-author">OmniRAG Enterprise Engine</div>
      <div class="message-content"><span class="pulse-dot"></span> Synthesizing grounded context...</div>
    </div>
  `;
  const contentEl = container.querySelector('.message-content');
  return { container, contentEl };
}

function appendSystemNotification(html) {
  const notif = document.createElement('div');
  notif.className = 'chat-message bot-message';
  notif.style.opacity = '0.85';
  notif.innerHTML = `
    <div class="message-avatar" style="background: rgba(255, 255, 255, 0.05); border-color: var(--border-subtle); color: var(--text-muted);">
      <span>SYS</span>
    </div>
    <div class="message-body">
      <div class="message-content" style="padding: 10px 14px; font-size: 0.8rem;">${html}</div>
    </div>
  `;
  chatHistory.appendChild(notif);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function renderStreamingMarkdown(element, rawText) {
  let parsedHtml = marked.parse(rawText);

  // Convert inline citations like [1], [2] into interactive badges
  parsedHtml = parsedHtml.replace(/\[(\d+)\]/g, (match, p1) => {
    return `<span class="citation-badge" data-index="${p1}">[${p1}]</span>`;
  });

  element.innerHTML = parsedHtml;

  // Add event listeners to citation badges
  element.querySelectorAll('.citation-badge').forEach(badge => {
    badge.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(badge.dataset.index);
      const src = state.activeSources.find(s => s.citation_index === idx);
      if (src) {
        openProvenanceModal(src);
      }
    });
  });
}

function openProvenanceModal(src) {
  document.getElementById('modal-citation-badge').textContent = `[${src.citation_index}]`;
  document.getElementById('modal-doc-title').textContent = src.doc_title;
  document.getElementById('modal-section').textContent = src.section_header || 'General Overview';
  document.getElementById('modal-page').textContent = src.page_number || '1';
  document.getElementById('modal-method').textContent = src.retrieval_method || 'Hybrid (Dense + BM25)';
  document.getElementById('modal-score').textContent = `${(src.rerank_score * 100).toFixed(1)}% (Cross-Encoder Attention)`;
  document.getElementById('modal-chunk-text').textContent = src.full_content;

  provenanceModal.classList.remove('hidden');
}

function escapeHTML(str) {
  if (!str) return '';
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}
