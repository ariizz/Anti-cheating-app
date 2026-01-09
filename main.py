from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import json
import time

app = FastAPI()

class IncidentModel(BaseModel):
    type: str
    details: dict = {}
    timestamp: Optional[str] = None  # We will set this if not provided

LOG_FILE = Path("incidents.log")

@app.post("/active_alert")
async def receive_alert(incident: IncidentModel):
    """Receive an alert from the computer vision client."""
    if not incident.timestamp:
        # Use current UTC time if not provided, passing it as string
        incident.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        
    # Append to log file
    entry = incident.dict()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    
    return {"status": "received"}


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
    """Professional dashboard that polls /incidents and shows them in a table."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Incidents Data</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; }
            .fade-in { animation: fadeIn 0.3s ease-in-out; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    </head>
    <body class="bg-gray-50 text-gray-800 h-screen flex flex-col overflow-hidden">
        
        <!-- Header -->
        <header class="bg-white border-b border-gray-200 px-8 py-4 flex justify-between items-center shadow-sm z-10">
            <div class="flex items-center gap-3">
                <div class="h-8 w-8 bg-black rounded-lg flex items-center justify-center">
                    <span class="text-white font-bold text-lg">I</span>
                </div>
                <h1 class="font-semibold text-xl tracking-tight text-gray-900">Incidents Data</h1>
            </div>
            <div class="flex items-center gap-4">
                <div class="flex items-center gap-2 text-sm text-gray-500 bg-gray-100 px-3 py-1.5 rounded-full">
                    <span class="relative flex h-2 w-2">
                      <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                      <span class="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                    </span>
                    Live Monitoring
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="flex-1 flex overflow-hidden">
            
            <!-- Sidebar / Stats -->
            <aside class="w-80 bg-white border-r border-gray-200 flex flex-col p-6 gap-6 overflow-y-auto hidden md:flex">
                <div>
                    <h2 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Overview</h2>
                    <div class="grid grid-cols-1 gap-4">
                        <div class="p-4 rounded-xl border border-gray-100 bg-gray-50">
                            <div class="text-2xl font-bold text-gray-900" id="total-incidents">0</div>
                            <div class="text-sm text-gray-500 font-medium">Total Incidents</div>
                        </div>
                        <div class="p-4 rounded-xl border border-red-50 bg-red-50/50">
                            <div class="text-2xl font-bold text-red-600" id="looking-away-count">0</div>
                            <div class="text-sm text-red-600/70 font-medium">Looking Away</div>
                        </div>
                         <div class="p-4 rounded-xl border border-orange-50 bg-orange-50/50">
                            <div class="text-2xl font-bold text-orange-600" id="lip-count">0</div>
                            <div class="text-sm text-orange-600/70 font-medium">Lip Movement</div>
                        </div>
                        <div class="p-4 rounded-xl border border-gray-100 bg-gray-50">
                            <div class="text-2xl font-bold text-gray-700" id="face-missing-count">0</div>
                            <div class="text-sm text-gray-500 font-medium">Face Missing</div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-auto">
                    <div class="text-xs text-gray-400 text-center">
                        Last updated: <span id="last-updated">Never</span>
                    </div>
                </div>
            </aside>

            <!-- Incident Feed -->
            <section class="flex-1 flex flex-col overflow-hidden bg-gray-50/50">
                <div class="px-8 py-6 border-b border-gray-200 bg-white/50 backdrop-blur-sm sticky top-0 z-10 flex justify-between items-center">
                    <h2 class="text-lg font-medium text-gray-900">Incident Log</h2>
                    <button onclick="loadIncidents()" class="text-sm text-blue-600 hover:text-blue-800 font-medium transition-colors">
                        Refresh Now
                    </button>
                </div>
                
                <div class="flex-1 overflow-y-auto px-8 py-6">
                    <div class="max-w-4xl mx-auto">
                        <table class="w-full text-left border-separate border-spacing-0">
                            <thead>
                                <tr>
                                    <th class="pb-4 font-medium text-gray-500 text-sm border-b border-gray-200 w-32">Time</th>
                                    <th class="pb-4 font-medium text-gray-500 text-sm border-b border-gray-200 w-48">Type</th>
                                    <th class="pb-4 font-medium text-gray-500 text-sm border-b border-gray-200">Details</th>
                                </tr>
                            </thead>
                            <tbody id="table-body" class="text-sm">
                                <!-- Incidents will be inserted here -->
                                <tr>
                                    <td colspan="3" class="py-12 text-center text-gray-400">
                                        Loading data...
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>
        </main>

        <script>
            function formatTime(timeStr) {
                if (!timeStr) return '-';
                // Try to parse string to get only H:M:S
                try {
                    return timeStr.split(' ')[1]; 
                } catch(e) { return timeStr; }
            }

            async function loadIncidents() {
                try {
                    const res = await fetch('/incidents');
                    const data = await res.json();
                    
                    // Stats
                    const recentData = data.reverse(); // Show newest first
                    const total = data.length;
                    const lookingAway = data.filter(i => i.type === 'LOOKING_AWAY').length;
                    const faceMissing = data.filter(i => i.type === 'FACE_NOT_VISIBLE').length;
                    const lipMovement = data.filter(i => i.type === 'LIP_MOVEMENT').length;
                    
                    document.getElementById('total-incidents').textContent = total;
                    document.getElementById('looking-away-count').textContent = lookingAway;
                    document.getElementById('face-missing-count').textContent = faceMissing;
                    document.getElementById('lip-count').textContent = lipMovement;
                    
                    const now = new Date();
                    document.getElementById('last-updated').textContent = now.toLocaleTimeString();

                    const tbody = document.getElementById('table-body');
                    tbody.innerHTML = '';

                    if (recentData.length === 0) {
                        tbody.innerHTML = `
                            <tr>
                                <td colspan="3" class="py-20 text-center">
                                    <div class="inline-flex items-center justify-center w-12 h-12 bg-green-100 rounded-full mb-3">
                                        <svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                                    </div>
                                    <p class="text-gray-500 font-medium">All clear</p>
                                    <p class="text-gray-400 text-xs mt-1">No incidents recorded yet.</p>
                                </td>
                            </tr>
                        `;
                        return;
                    }

                    recentData.forEach((inc) => {
                        const tr = document.createElement('tr');
                        tr.className = 'group hover:bg-white transition-colors fade-in';
                        
                        // Type Badge Style
                        let typeClass = 'bg-gray-100 text-gray-600';
                        let typeLabel = inc.type;
                        
                        if (inc.type === 'LOOKING_AWAY') {
                            typeClass = 'bg-yellow-100 text-yellow-700 border border-yellow-200';
                            typeLabel = 'Looking Away';
                        } else if (inc.type === 'FACE_NOT_VISIBLE') {
                            typeClass = 'bg-red-100 text-red-700 border border-red-200';
                            typeLabel = 'Face Missing';
                        } else if (inc.type === 'LIP_MOVEMENT' || inc.type === 'Lip movement detected') { // Handle both just in case
                             typeClass = 'bg-orange-100 text-orange-700 border border-orange-200';
                             typeLabel = 'Lip Movement';
                        }

                        // Details Check
                        let details = inc.details.reason || JSON.stringify(inc.details);
                        if (details === '{}') details = '';

                        tr.innerHTML = `
                            <td class="py-4 border-b border-gray-100 text-gray-500 font-mono text-xs group-hover:text-gray-700">
                                ${formatTime(inc.timestamp)}
                            </td>
                            <td class="py-4 border-b border-gray-100">
                                <span class="inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium ${typeClass}">
                                    ${typeLabel}
                                </span>
                            </td>
                            <td class="py-4 border-b border-gray-100 text-gray-600">
                                ${details}
                            </td>
                        `;
                        tbody.appendChild(tr);
                    });

                } catch (e) {
                    console.error("Fetch error", e);
                }
            }

            // Auto refresh every 2s
            setInterval(loadIncidents, 2000);
            loadIncidents();
        </script>
    </body>
    </html>
    """
    return html

