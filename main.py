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
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Exam Proctoring Dashboard</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
      <style>
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
          padding: 20px;
          color: #333;
        }

        .container {
          max-width: 1400px;
          margin: 0 auto;
        }

        .header {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          border-radius: 20px;
          padding: 30px 40px;
          margin-bottom: 30px;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 20px;
        }

        .header-content h1 {
          font-size: 32px;
          font-weight: 700;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-bottom: 8px;
        }

        .header-content p {
          color: #666;
          font-size: 15px;
          font-weight: 400;
        }

        .stats-container {
          display: flex;
          gap: 15px;
          flex-wrap: wrap;
        }

        .stat-card {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 15px 25px;
          border-radius: 12px;
          min-width: 150px;
          text-align: center;
          box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .stat-value {
          font-size: 28px;
          font-weight: 700;
          margin-bottom: 5px;
        }

        .stat-label {
          font-size: 12px;
          opacity: 0.9;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .dashboard-card {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          border-radius: 20px;
          padding: 30px;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
          overflow: hidden;
        }

        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 25px;
          padding-bottom: 20px;
          border-bottom: 2px solid #f0f0f0;
        }

        .card-title {
          font-size: 24px;
          font-weight: 600;
          color: #2d3748;
        }

        .refresh-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
          color: #667eea;
          font-size: 14px;
          font-weight: 500;
        }

        .refresh-dot {
          width: 8px;
          height: 8px;
          background: #667eea;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.2); }
        }

        .table-wrapper {
          overflow-x: auto;
          border-radius: 12px;
        }

        table {
          width: 100%;
          border-collapse: separate;
          border-spacing: 0;
          background: white;
        }

        thead {
          background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        }

        th {
          padding: 18px 20px;
          text-align: left;
          font-weight: 600;
          font-size: 13px;
          color: #4a5568;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          border-bottom: 2px solid #e2e8f0;
        }

        th:first-child {
          border-top-left-radius: 12px;
        }

        th:last-child {
          border-top-right-radius: 12px;
        }

        td {
          padding: 18px 20px;
          border-bottom: 1px solid #f0f0f0;
          font-size: 14px;
          color: #4a5568;
        }

        tbody tr {
          transition: all 0.3s ease;
          cursor: pointer;
        }

        tbody tr:hover {
          background: #f8f9fa;
          transform: translateX(4px);
          box-shadow: -4px 0 0 #667eea;
        }

        tbody tr:last-child td:first-child {
          border-bottom-left-radius: 12px;
        }

        tbody tr:last-child td:last-child {
          border-bottom-right-radius: 12px;
        }

        .LOOKING_AWAY {
          background: linear-gradient(90deg, #fff8e1 0%, #fffef5 100%);
          border-left: 4px solid #ffc107;
        }

        .FACE_NOT_VISIBLE {
          background: linear-gradient(90deg, #ffebee 0%, #fff5f6 100%);
          border-left: 4px solid #dc3545;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          padding: 6px 14px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .badge-looking {
          background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
          color: #212529;
        }

        .badge-missing {
          background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
          color: #fff;
        }

        .time-cell {
          font-family: 'Courier New', monospace;
          font-weight: 500;
          color: #667eea;
        }

        .details-cell {
          font-family: 'Courier New', monospace;
          font-size: 12px;
          color: #718096;
          background: #f7fafc;
          padding: 10px 15px;
          border-radius: 8px;
          max-width: 400px;
          word-break: break-all;
        }

        .empty-state {
          text-align: center;
          padding: 60px 20px;
          color: #a0aec0;
        }

        .empty-state-icon {
          font-size: 64px;
          margin-bottom: 20px;
          opacity: 0.5;
        }

        .empty-state-text {
          font-size: 18px;
          font-weight: 500;
          margin-bottom: 8px;
        }

        .empty-state-subtext {
          font-size: 14px;
          color: #cbd5e0;
        }

        .loading {
          text-align: center;
          padding: 40px;
          color: #667eea;
        }

        .spinner {
          border: 3px solid #f3f3f3;
          border-top: 3px solid #667eea;
          border-radius: 50%;
          width: 40px;
          height: 40px;
          animation: spin 1s linear infinite;
          margin: 0 auto 15px;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
          .header {
            flex-direction: column;
            text-align: center;
          }

          .stats-container {
            justify-content: center;
          }

          .table-wrapper {
            overflow-x: scroll;
          }

          th, td {
            padding: 12px 15px;
            font-size: 13px;
          }
        }

        .fade-in {
          animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <div class="header-content">
            <h1>📊 Exam Proctoring Dashboard</h1>
            <p>Real-time monitoring of exam incidents and violations</p>
          </div>
          <div class="stats-container">
            <div class="stat-card">
              <div class="stat-value" id="total-incidents">0</div>
              <div class="stat-label">Total Incidents</div>
            </div>
            <div class="stat-card">
              <div class="stat-value" id="looking-away-count">0</div>
              <div class="stat-label">Looking Away</div>
            </div>
            <div class="stat-card">
              <div class="stat-value" id="face-missing-count">0</div>
              <div class="stat-label">Face Missing</div>
            </div>
          </div>
        </div>

        <div class="dashboard-card">
          <div class="card-header">
            <h2 class="card-title">📋 Incident Log</h2>
            <div class="refresh-indicator">
              <div class="refresh-dot"></div>
              <span>Auto-refreshing every 2s</span>
            </div>
          </div>

          <div class="table-wrapper">
            <table id="incidents-table">
              <thead>
                <tr>
                  <th>⏰ Time (UTC)</th>
                  <th>🏷️ Type</th>
                  <th>📝 Details</th>
                </tr>
              </thead>
              <tbody id="table-body">
                <tr>
                  <td colspan="3" class="loading">
                    <div class="spinner"></div>
                    <div>Loading incidents...</div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <script>
        async function loadIncidents() {
          try {
            const res = await fetch('/incidents');
            const data = await res.json();
            const tbody = document.querySelector('#table-body');
            
            // Update statistics
            const totalIncidents = data.length;
            const lookingAwayCount = data.filter(inc => inc.type === 'LOOKING_AWAY').length;
            const faceMissingCount = data.filter(inc => inc.type === 'FACE_NOT_VISIBLE').length;
            
            document.getElementById('total-incidents').textContent = totalIncidents;
            document.getElementById('looking-away-count').textContent = lookingAwayCount;
            document.getElementById('face-missing-count').textContent = faceMissingCount;
            
            // Clear table
            tbody.innerHTML = '';
            
            if (data.length === 0) {
              tbody.innerHTML = `
                <tr>
                  <td colspan="3" class="empty-state">
                    <div class="empty-state-icon">✅</div>
                    <div class="empty-state-text">No incidents detected</div>
                    <div class="empty-state-subtext">All clear! No violations recorded.</div>
                  </td>
                </tr>
              `;
              return;
            }
            
            // Add incidents to table
            data.forEach((inc, index) => {
              const tr = document.createElement('tr');
              tr.className = (inc.type || '') + ' fade-in';
              tr.style.animationDelay = `${index * 0.05}s`;

              const tdTime = document.createElement('td');
              const tdType = document.createElement('td');
              const tdDetails = document.createElement('td');

              tdTime.textContent = inc.timestamp || 'N/A';
              tdTime.className = 'time-cell';

              const typeSpan = document.createElement('span');
              if (inc.type === 'LOOKING_AWAY') {
                typeSpan.textContent = '👀 Looking Away';
                typeSpan.className = 'badge badge-looking';
              } else if (inc.type === 'FACE_NOT_VISIBLE') {
                typeSpan.textContent = '🚫 Face Not Visible';
                typeSpan.className = 'badge badge-missing';
              } else {
                typeSpan.textContent = inc.type || 'Unknown';
                typeSpan.className = 'badge';
              }
              tdType.appendChild(typeSpan);

              const detailsText = JSON.stringify(inc.details || {}, null, 2);
              tdDetails.textContent = detailsText;
              tdDetails.className = 'details-cell';

              tr.appendChild(tdTime);
              tr.appendChild(tdType);
              tr.appendChild(tdDetails);
              tbody.appendChild(tr);
            });
          } catch (e) {
            console.error('Failed to load incidents', e);
            const tbody = document.querySelector('#table-body');
            tbody.innerHTML = `
              <tr>
                <td colspan="3" class="empty-state">
                  <div class="empty-state-icon">⚠️</div>
                  <div class="empty-state-text">Error loading incidents</div>
                  <div class="empty-state-subtext">${e.message}</div>
                </td>
              </tr>
            `;
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

