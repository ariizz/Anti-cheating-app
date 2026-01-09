from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
import json

app = FastAPI()

LOG_FILE = Path("incidents.log")


@app.get("/incidents")
def get_incidents():
    """Return all logged incidents as a JSON list."""
    if not LOG_FILE.exists():
        return []

    incidents = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                incidents.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip any corrupted lines
                continue
    return incidents


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Very simple dashboard that polls /incidents and shows them in a table."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Exam Proctoring Dashboard</title>
      <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #f4f5f7; }
        h1 { margin-bottom: 0.5rem; }
        p { margin-top: 0; color: #555; }
        table { border-collapse: collapse; width: 100%; background: #fff; margin-top: 1rem; }
        th, td { border: 1px solid #e0e0e0; padding: 8px 10px; text-align: left; font-size: 14px; }
        th { background: #fafafa; font-weight: 600; }
        tr:nth-child(even) { background: #fafafa; }
        .LOOKING_AWAY { background-color: #fff8e1; }
        .FACE_NOT_VISIBLE { background-color: #ffebee; }
        .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
        .badge-looking { background: #ffc107; color: #212529; }
        .badge-missing { background: #dc3545; color: #fff; }
      </style>
    </head>
    <body>
      <h1>Exam Proctoring Dashboard</h1>
      <p>Incidents are generated locally by the proctoring script and streamed here in real time.</p>
      <table id="incidents-table">
        <thead>
          <tr>
            <th>Time (UTC)</th>
            <th>Type</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>

      <script>
        async function loadIncidents() {
          try {
            const res = await fetch('/incidents');
            const data = await res.json();
            const tbody = document.querySelector('#incidents-table tbody');
            tbody.innerHTML = '';
            data.forEach(inc => {
              const tr = document.createElement('tr');
              tr.className = inc.type || '';

              const tdTime = document.createElement('td');
              const tdType = document.createElement('td');
              const tdDetails = document.createElement('td');

              tdTime.textContent = inc.timestamp || '';

              const typeSpan = document.createElement('span');
              if (inc.type === 'LOOKING_AWAY') {
                typeSpan.textContent = 'Looking Away';
                typeSpan.className = 'badge badge-looking';
              } else if (inc.type === 'FACE_NOT_VISIBLE') {
                typeSpan.textContent = 'Face Not Visible';
                typeSpan.className = 'badge badge-missing';
              } else {
                typeSpan.textContent = inc.type || '';
                typeSpan.className = 'badge';
              }
              tdType.appendChild(typeSpan);

              tdDetails.textContent = JSON.stringify(inc.details || {});

              tr.appendChild(tdTime);
              tr.appendChild(tdType);
              tr.appendChild(tdDetails);
              tbody.appendChild(tr);
            });
          } catch (e) {
            console.error('Failed to load incidents', e);
          }
        }

        // Poll every 2 seconds
        setInterval(loadIncidents, 2000);
        loadIncidents();
      </script>
    </body>
    </html>
    """
    return html

