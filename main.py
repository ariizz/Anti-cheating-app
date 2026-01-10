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
    """Premium Dark Dashboard with Glassmorphism for AI Proctoring."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Drishti AI | Proctoring Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --accent: #3b82f6;
                --accent-glow: rgba(59, 130, 246, 0.5);
                --bg: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --border: rgba(255, 255, 255, 0.08);
            }
            body { 
                font-family: 'Outfit', sans-serif;
                background-color: var(--bg);
                background-image: 
                    radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(147, 51, 234, 0.1) 0px, transparent 50%);
                color: #f8fafc;
            }
            .glass {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--border);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            }
            .incident-row {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .incident-row:hover {
                background: rgba(255, 255, 255, 0.03);
            }
            @keyframes pulse-custom {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(1.1); }
            }
            .pulse-green { animation: pulse-custom 2s infinite; }
            
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
            ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
            ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
        </style>
    </head>
    <body class="h-screen flex flex-col overflow-hidden">
        
        <nav class="glass border-b border-white/5 px-8 py-4 flex justify-between items-center z-50">
            <div class="flex items-center gap-4">
                <div class="h-10 w-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                    <span class="text-white font-bold text-xl">¬_¬</span>
                </div>
                <div>
                    <h1 class="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">Drishti AI</h1>
                    <p class="text-[10px] uppercase tracking-[0.2em] text-blue-400 font-bold">Proctoring Control Center</p>
                </div>
            </div>
            <div class="flex items-center gap-6">
                <div class="flex items-center gap-3 bg-white/5 border border-white/10 px-4 py-2 rounded-full">
                    <div class="h-2 w-2 rounded-full bg-green-500 pulse-green"></div>
                    <span class="text-sm font-medium text-white/80">System Live</span>
                </div>
                <div class="text-sm text-white/50" id="current-time">00:00:00</div>
            </div>
        </nav>

        <main class="flex-1 flex overflow-hidden p-6 gap-6">
            <aside class="w-80 flex flex-col gap-6">
                <div class="glass rounded-3xl p-6 flex flex-col gap-4">
                    <h2 class="text-xs font-bold text-blue-400 uppercase tracking-widest">Analytics</h2>
                    <div class="grid grid-cols-1 gap-4">
                        <div class="bg-white/5 rounded-2xl p-4 border border-white/5">
                            <div class="text-3xl font-bold text-white" id="total-incidents">0</div>
                            <div class="text-xs text-white/40 mt-1 uppercase tracking-wider">Total Events</div>
                        </div>
                    </div>
                </div>

                <div class="glass rounded-3xl p-6 flex-1 overflow-y-auto">
                    <h2 class="text-xs font-bold text-blue-400 uppercase tracking-widest mb-6">Breach Summary</h2>
                    <div class="space-y-6">
                        <div class="space-y-2">
                            <div class="flex justify-between text-[11px] uppercase tracking-wider font-bold text-white/60">
                                <span>Looking Away</span>
                                <span id="looking-away-count">0</span>
                            </div>
                            <div class="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                <div id="looking-away-bar" class="h-full bg-blue-500 transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="space-y-2">
                            <div class="flex justify-between text-[11px] uppercase tracking-wider font-bold text-white/60">
                                <span>Lip Movement</span>
                                <span id="lip-count">0</span>
                            </div>
                            <div class="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                <div id="lip-bar" class="h-full bg-purple-500 transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="space-y-2">
                            <div class="flex justify-between text-[11px] uppercase tracking-wider font-bold text-white/60">
                                <span>Face Missing</span>
                                <span id="face-missing-count">0</span>
                            </div>
                            <div class="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                <div id="face-missing-bar" class="h-full bg-red-500 transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="space-y-2">
                            <div class="flex justify-between text-[11px] uppercase tracking-wider font-bold text-white/60">
                                <span>Multiple Faces</span>
                                <span id="multi-face-count">0</span>
                            </div>
                            <div class="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                <div id="multi-face-bar" class="h-full bg-pink-500 transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </aside>

            <section class="flex-1 flex flex-col gap-6">
                <div class="glass rounded-[2rem] flex flex-col overflow-hidden">
                    <div class="px-8 py-6 border-b border-white/5 backdrop-blur-md flex justify-between items-center">
                        <h2 class="text-lg font-bold">Real-time Stream</h2>
                        <button onclick="loadIncidents()" class="bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold px-5 py-2.5 rounded-xl transition-all active:scale-95">
                            Refresh
                        </button>
                    </div>
                    
                    <div class="flex-1 overflow-y-auto" style="height: calc(100vh - 220px);">
                        <table class="w-full text-left border-collapse">
                            <thead class="sticky top-0 bg-[#162035] text-[10px] uppercase tracking-[0.2em] text-white/40 font-bold">
                                <tr>
                                    <th class="px-8 py-4 border-b border-white/5">Time</th>
                                    <th class="px-4 py-4 border-b border-white/5">Incident Type</th>
                                    <th class="px-8 py-4 border-b border-white/5">Details</th>
                                </tr>
                            </thead>
                            <tbody id="table-body" class="text-sm">
                                <tr>
                                    <td colspan="3" class="py-24 text-center opacity-30 text-xs tracking-widest uppercase">
                                        Initializing Stream...
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>
        </main>

        <script>
            function updateClock() {
                const now = new Date();
                document.getElementById('current-time').textContent = now.toLocaleTimeString();
            }
            setInterval(updateClock, 1000);
            updateClock();

            function formatTime(timeStr) {
                if (!timeStr) return '-';
                try { return timeStr.split(' ')[1]; } catch(e) { return timeStr; }
            }

            async function loadIncidents() {
                try {
                    const res = await fetch('/incidents');
                    const data = await res.json();
                    const recentData = [...data].reverse();
                    
                    const lookingAway = data.filter(i => i.type === 'LOOKING_AWAY').length;
                    const faceMissing = data.filter(i => i.type === 'FACE_NOT_VISIBLE').length;
                    const lipMovement = data.filter(i => i.type === 'LIP_MOVEMENT').length;
                    const multiFace = data.filter(i => i.type === 'MULTIPLE_FACES').length;

                    document.getElementById('total-incidents').textContent = data.length;
                    document.getElementById('looking-away-count').textContent = lookingAway;
                    document.getElementById('face-missing-count').textContent = faceMissing;
                    document.getElementById('lip-count').textContent = lipMovement;
                    document.getElementById('multi-face-count').textContent = multiFace;

                    const max = Math.max(lookingAway, faceMissing, lipMovement, multiFace, 1);
                    document.getElementById('looking-away-bar').style.width = (lookingAway/max*100) + '%';
                    document.getElementById('face-missing-bar').style.width = (faceMissing/max*100) + '%';
                    document.getElementById('lip-bar').style.width = (lipMovement/max*100) + '%';
                    document.getElementById('multi-face-bar').style.width = (multiFace/max*100) + '%';

                    const tbody = document.getElementById('table-body');
                    tbody.innerHTML = '';

                    if (recentData.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="3" class="py-32 text-center opacity-30 uppercase text-xs tracking-widest font-bold">Secure Zone: No alerts</td></tr>';
                        return;
                    }

                    recentData.forEach((inc) => {
                        const tr = document.createElement('tr');
                        tr.className = 'incident-row border-b border-white/[0.02]';
                        
                        let badge = 'bg-white/10 text-white';
                        if (inc.type === 'LOOKING_AWAY') badge = 'bg-blue-500/10 text-blue-400 border border-blue-500/20';
                        else if (inc.type === 'FACE_NOT_VISIBLE') badge = 'bg-red-500/10 text-red-400 border border-red-500/20';
                        else if (inc.type === 'LIP_MOVEMENT') badge = 'bg-purple-500/10 text-purple-400 border border-purple-500/20';
                        else if (inc.type === 'MULTIPLE_FACES') badge = 'bg-pink-500/10 text-pink-400 border border-pink-500/20';

                        tr.innerHTML = `
                            <td class="px-8 py-4 font-mono text-xs text-white/40">
                                ${formatTime(inc.timestamp)}
                            </td>
                            <td class="px-4 py-4">
                                <span class="px-2 py-1 rounded-md text-[10px] font-bold uppercase ${badge}">
                                    ${inc.type.replace(/_/g, ' ')}
                                </span>
                            </td>
                            <td class="px-8 py-4 text-white/60">
                                ${inc.details.reason || ''}
                            </td>
                        `;
                        tbody.appendChild(tr);
                    });
                } catch (e) { console.error(e); }
            }

            setInterval(loadIncidents, 2000);
            loadIncidents();
        </script>
    </body>
    </html>
    """
    return html

