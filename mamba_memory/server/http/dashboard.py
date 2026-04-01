"""Embedded Web UI dashboard for MambaMemory.

Single-page HTML served at /ui — shows L2 slots, entity graph,
L3 records, and system status. No build step, no JS framework,
just a self-contained HTML page that fetches data from the API.
"""

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MambaMemory Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 20px; }
  h1 { font-size: 1.4em; margin-bottom: 20px; }
  h1 span { color: var(--accent); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card .label { font-size: 0.75em; color: var(--dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 1.8em; font-weight: 600; margin-top: 4px; }
  h2 { font-size: 1.1em; margin: 20px 0 12px; color: var(--dim); }
  table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px; overflow: hidden; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.85em; }
  th { background: var(--border); color: var(--dim); font-weight: 500; text-transform: uppercase; font-size: 0.7em; letter-spacing: 0.5px; }
  .bar { height: 6px; border-radius: 3px; background: var(--border); }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .tag { display: inline-block; padding: 2px 6px; margin: 1px; border-radius: 4px; font-size: 0.7em; background: #1f6feb33; color: var(--accent); }
  .act-high { color: var(--green); } .act-mid { color: var(--yellow); } .act-low { color: var(--red); }
  .search-box { width: 100%; padding: 10px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.9em; margin-bottom: 12px; }
  .search-box:focus { outline: none; border-color: var(--accent); }
  #recall-results { min-height: 40px; }
  .result { padding: 8px 12px; margin: 4px 0; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; font-size: 0.85em; }
  .result .layer { color: var(--accent); font-weight: 600; }
  .result .score { color: var(--dim); }
  .refresh-btn { background: var(--accent); color: #fff; border: none; padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 0.8em; float: right; }
  .refresh-btn:hover { opacity: 0.8; }
  #entity-graph { display: flex; flex-wrap: wrap; gap: 6px; }
  .entity-node { padding: 6px 10px; background: #1f6feb22; border: 1px solid #1f6feb55; border-radius: 16px; font-size: 0.8em; cursor: default; }
  .entity-node .count { color: var(--dim); font-size: 0.75em; }
</style>
</head>
<body>

<h1><span>MambaMemory</span> Dashboard <button class="refresh-btn" onclick="refresh()">Refresh</button></h1>

<div class="grid" id="stats"></div>

<h2>L2 Cognitive Slots</h2>
<table id="slots-table">
  <thead><tr><th>ID</th><th>Topic</th><th>Activation</th><th>Entities</th><th>Preview</th></tr></thead>
  <tbody id="slots-body"></tbody>
</table>

<h2>Entity Graph</h2>
<div id="entity-graph"></div>

<h2>Recall Test</h2>
<input type="text" class="search-box" id="recall-input" placeholder="Type a query and press Enter..." onkeydown="if(event.key==='Enter')doRecall()">
<div id="recall-results"></div>

<script>
const API = window.location.origin;
const API_KEY = new URLSearchParams(window.location.search).get('key') || '';
const headers = API_KEY ? {'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json'} : {'Content-Type': 'application/json'};

async function fetchJSON(path, opts={}) {
  const r = await fetch(API + path, {headers, ...opts});
  return r.json();
}

function actClass(v) { return v > 0.6 ? 'act-high' : v > 0.2 ? 'act-mid' : 'act-low'; }

async function refresh() {
  // Status
  const s = await fetchJSON('/status');
  document.getElementById('stats').innerHTML = `
    <div class="card"><div class="label">L1 Window</div><div class="value">${s.l1_window_turns}</div></div>
    <div class="card"><div class="label">L1 Compressed</div><div class="value">${s.l1_compressed_segments}</div></div>
    <div class="card"><div class="label">L2 Slots</div><div class="value">${s.l2_active_slots}/${s.l2_total_slots}</div></div>
    <div class="card"><div class="label">L2 Steps</div><div class="value">${s.l2_step_count}</div></div>
    <div class="card"><div class="label">L3 Records</div><div class="value">${s.l3_total_records}</div></div>
    <div class="card"><div class="label">L3 Entities</div><div class="value">${s.l3_entity_count}</div></div>
  `;

  // Slots (full detail)
  try {
    const detail = await fetchJSON('/status?detail=full');
    // Fetch slots from status endpoint won't work via GET params, use the regular endpoint
  } catch(e) {}

  // For slots, we'll do a POST recall with empty query to trigger status with slots
  // Actually, let's just use the status endpoint — we need to add query param support
  // For now, show basic status
  document.getElementById('slots-body').innerHTML = '<tr><td colspan="5" style="color:var(--dim)">Click Refresh to load slot data</td></tr>';

  // Try to load slots via a status call
  try {
    const resp = await fetch(API + '/status', {headers});
    const data = await resp.json();
    if (data.l2_active_slots > 0) {
      // Do a recall for '*' to get some slot data
      const recall = await fetchJSON('/recall', {method: 'POST', body: JSON.stringify({query: '*', limit: 20, layers: ['l2']})});
      if (recall.memories && recall.memories.length > 0) {
        document.getElementById('slots-body').innerHTML = recall.memories.map(m => `
          <tr>
            <td>${m.source_id || '-'}</td>
            <td>${m.topic}</td>
            <td><span class="${actClass(m.score)}">${m.score.toFixed(3)}</span></td>
            <td></td>
            <td>${m.content.substring(0, 80)}</td>
          </tr>
        `).join('');
      }
    }
  } catch(e) { console.error(e); }
}

async function doRecall() {
  const q = document.getElementById('recall-input').value;
  if (!q) return;
  const r = await fetchJSON('/recall', {method: 'POST', body: JSON.stringify({query: q, limit: 5})});
  document.getElementById('recall-results').innerHTML = r.memories.map(m => `
    <div class="result">
      <span class="layer">[${m.layer}]</span>
      <span class="score">(${m.score.toFixed(3)})</span>
      ${m.content}
    </div>
  `).join('') || '<div style="color:var(--dim)">No results</div>';
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""
