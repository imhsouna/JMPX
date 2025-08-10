#!/usr/bin/env python3
import io
import os
import json
import time
import queue
import threading
from collections import deque
from typing import Optional

from flask import Flask, Response, render_template_string, request, redirect, url_for, jsonify
import numpy as np
import sounddevice as sd

from rds2_stream import (
    RdsConfig,
    RdsBitstreamGenerator,
    make_mpx,
    read_audio_file,
    generate_tone,
    RDS_BITRATE,
    load_logo_bits,
)

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
  <title>FM MPX + RDS/RDS2</title>
  <style>
    .logbox { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; white-space: pre-wrap; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800">
  <div class="max-w-6xl mx-auto p-6 space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold">FM MPX + RDS/RDS2 Web UI</h1>
      <div class="flex items-center gap-3">
        <span id="statusBadge" class="px-3 py-1 rounded text-sm bg-gray-200 text-gray-800">Loading…</span>
        <button id="stopBtn" class="hidden bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded">Stop</button>
      </div>
    </div>

    <form id="startForm" class="grid grid-cols-1 md:grid-cols-2 gap-4" method="post" enctype="multipart/form-data" action="{{ url_for('start') }}">
      <div class="space-y-3 p-4 bg-white rounded shadow">
        <h2 class="font-semibold">Audio Source</h2>
        <div>
          <label class="block text-sm">Sample Rate (Hz)</label>
          <input class="border rounded px-2 py-1 w-full" type="number" name="fs" value="192000" />
        </div>
        <div>
          <label class="inline-flex items-center gap-2">
            <input type="radio" name="source" value="tone" checked /> Tone
          </label>
          <label class="inline-flex items-center gap-2 ml-4">
            <input type="radio" name="source" value="file" /> File
          </label>
        </div>
        <div id="toneRow">
          <label class="block text-sm">Tone (Hz)</label>
          <input class="border rounded px-2 py-1 w-full" type="number" step="1" name="tone" value="1000" />
          <label class="block text-sm mt-2">Duration (s)</label>
          <input class="border rounded px-2 py-1 w-full" type="number" step="1" name="duration" value="3600" />
        </div>
        <div id="fileRow" class="hidden">
          <label class="block text-sm">Upload audio file</label>
          <input class="border rounded px-2 py-1 w-full" type="file" name="audio" accept="audio/*" />
        </div>
        <div>
          <label class="block text-sm">Output Device</label>
          <select class="border rounded px-2 py-1 w-full" name="device">
            <option value="">Default</option>
            {% for d in devices %}
              <option value="{{ d.index }}">{{ d.index }} - {{ d.name }}</option>
            {% endfor %}
          </select>
        </div>
      </div>

      <div class="space-y-3 p-4 bg-white rounded shadow">
        <h2 class="font-semibold">RDS / RDS2</h2>
        <div class="grid grid-cols-2 gap-2">
          <div>
            <label class="block text-sm">PI (hex)</label>
            <input class="border rounded px-2 py-1 w-full" name="pi" value="0x1234" />
          </div>
          <div>
            <label class="block text-sm">PS (name)</label>
            <input class="border rounded px-2 py-1 w-full" name="ps" value="RADIO" />
          </div>
        </div>
        <div>
          <label class="block text-sm">Radiotext</label>
          <input class="border rounded px-2 py-1 w-full" name="rt" value="Welcome to RADIO" />
        </div>
        <div class="grid grid-cols-3 gap-2">
          <div>
            <label class="block text-sm">Pilot level</label>
            <input class="border rounded px-2 py-1 w-full" name="pilot" value="0.08" />
          </div>
          <div>
            <label class="block text-sm">RDS level</label>
            <input class="border rounded px-2 py-1 w-full" name="rds" value="0.03" />
          </div>
          <div>
            <label class="block text-sm">RDS2 level</label>
            <input class="border rounded px-2 py-1 w-full" name="rds2" value="0.01" />
          </div>
        </div>
        <div>
          <label class="inline-flex items-center gap-2">
            <input type="checkbox" name="enable_rds2" checked /> Enable RDS2 (logo)
          </label>
        </div>
        <div>
          <label class="block text-sm">Upload Logo (png/jpg)</label>
          <input class="border rounded px-2 py-1 w-full" type="file" name="logo" accept="image/*" />
        </div>
      </div>

      <div class="md:col-span-2 flex items-center gap-3">
        <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded" type="submit">Start</button>
        <button id="stopBtn2" class="bg-gray-200 hover:bg-gray-300 px-4 py-2 rounded" type="button">Stop</button>
      </div>
    </form>

    <div class="p-4 bg-white rounded shadow">
      <h2 class="font-semibold mb-2">Live status</h2>
      <div id="liveStatus" class="text-sm text-slate-700">—</div>
    </div>

    <div class="p-4 bg-white rounded shadow">
      <div class="flex items-center justify-between mb-2">
        <h2 class="font-semibold">Logs</h2>
        <button id="clearLog" class="text-xs text-slate-500 hover:text-slate-700">Clear</button>
      </div>
      <div id="log" class="logbox bg-slate-100 rounded p-3 h-64 overflow-auto"></div>
    </div>
  </div>

  <script>
    const sourceRadios = document.querySelectorAll('input[name=source]');
    const toneRow = document.getElementById('toneRow');
    const fileRow = document.getElementById('fileRow');
    function updateSource(){
      const v = document.querySelector('input[name=source]:checked').value;
      toneRow.classList.toggle('hidden', v!=='tone');
      fileRow.classList.toggle('hidden', v!=='file');
    }
    sourceRadios.forEach(r=>r.addEventListener('change', updateSource));
    updateSource();

    const statusBadge = document.getElementById('statusBadge');
    const liveStatus = document.getElementById('liveStatus');
    const logBox = document.getElementById('log');
    const stopBtn = document.getElementById('stopBtn');
    const stopBtn2 = document.getElementById('stopBtn2');
    const clearLog = document.getElementById('clearLog');

    function setBadge(running){
      if(running){
        statusBadge.textContent = 'Running';
        statusBadge.className = 'px-3 py-1 rounded text-sm bg-green-600 text-white';
        stopBtn.classList.remove('hidden');
      }else{
        statusBadge.textContent = 'Stopped';
        statusBadge.className = 'px-3 py-1 rounded text-sm bg-gray-300 text-gray-800';
        stopBtn.classList.add('hidden');
      }
    }

    function appendLog(line){
      const atBottom = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 5;
      const div = document.createElement('div');
      div.textContent = line;
      logBox.appendChild(div);
      if(atBottom){ logBox.scrollTop = logBox.scrollHeight; }
    }

    stopBtn.onclick = stopBtn2.onclick = async () => {
      try{ await fetch('/stop', {method:'POST'}); }catch(e){}
    };
    clearLog.onclick = () => { logBox.innerHTML=''; };

    // Connect to SSE for status+logs
    const es = new EventSource('/events');
    es.onmessage = (ev) => {
      try{
        const msg = JSON.parse(ev.data);
        if(msg.type === 'status'){
          setBadge(!!msg.running);
          liveStatus.textContent = `Device: ${msg.device||'Default'} | fs: ${msg.fs||'-'} | Source: ${msg.source||'-'} | Uptime: ${msg.uptime_s||0}s`;
        }else if(msg.type === 'log'){
          appendLog(msg.text);
        }
      }catch(e){ /* ignore */ }
    };
  </script>
</body>
</html>
"""

app = Flask(__name__)

_stream_thread: Optional[threading.Thread] = None
_stop_flag = threading.Event()
_state_lock = threading.Lock()
STATE = {
    'running': False,
    'started_at': None,
    'device': None,
    'fs': None,
    'source': None,
    'last_error': None,
}

LOGS = deque(maxlen=500)
LOG_EVENT = threading.Event()


def log(line: str):
    ts = time.strftime('%H:%M:%S')
    entry = f"[{ts}] {line}"
    LOGS.append(entry)
    LOG_EVENT.set()


def list_output_devices():
    devs = []
    try:
        for idx, d in enumerate(sd.query_devices()):
            if d.get('max_output_channels', 0) > 0:
                devs.append({'index': idx, 'name': d['name']})
    except Exception:
        pass
    return devs


def run_stream(fs: int, device: Optional[int], audio_path: Optional[str], tone: Optional[float], duration: float,
               pi_hex: str, ps: str, rt: str, pilot_level: float, rds_level: float, rds2_level: float,
               enable_rds2: bool, logo_path: Optional[str]):
    try:
        sd.default.samplerate = fs
        if device is not None:
            sd.default.device = device
        dev_name = None
        try:
            dev_info = sd.query_devices(device) if device is not None else sd.query_devices(sd.default.device)
            dev_name = dev_info.get('name') if isinstance(dev_info, dict) else None
        except Exception:
            dev_name = None

        if audio_path:
            stereo, _ = read_audio_file(audio_path, target_fs=fs)
            source = 'file'
        else:
            stereo = generate_tone(duration_s=duration, fs=fs, freq_hz=tone or 1000.0)
            source = 'tone'

        cfg = RdsConfig(pi_code=int(pi_hex, 16), program_service_name=ps or '', radiotext=rt or '')
        gen = RdsBitstreamGenerator(cfg)
        if enable_rds2 and logo_path:
            gen.set_logo_bits(load_logo_bits(logo_path))

        total_seconds = stereo.shape[0] / fs
        rds_bits = gen.generate_bits(int(total_seconds * RDS_BITRATE * 1.1))

        blocksize = 4096
        idx = 0

        with _state_lock:
            STATE.update({'running': True, 'started_at': time.time(), 'device': dev_name, 'fs': fs, 'source': source, 'last_error': None})
        log(f"Starting stream: fs={fs}, device={dev_name or 'Default'}, source={source}, PS='{ps}', RDS2={'on' if enable_rds2 else 'off'}")

        def callback(outdata, frames, time_info, status):
            nonlocal idx, rds_bits
            if _stop_flag.is_set():
                raise sd.CallbackStop
            end = min(idx + frames, stereo.shape[0])
            left = stereo[idx:end, 0]
            right = stereo[idx:end, 1]
            bits_needed = int((end - idx) / fs * RDS_BITRATE) + 208
            if len(rds_bits) < bits_needed:
                extra = gen.generate_bits(int(2.0 * RDS_BITRATE))
                rds_bits = np.concatenate([rds_bits, extra])
            bits_block = rds_bits[:bits_needed]
            rds_bits = rds_bits[bits_needed:]
            mpx = make_mpx(left, right, fs, pilot_level, rds_level, rds2_level, bits_block, enable_rds2)
            if outdata.shape[1] == 1:
                outdata[:, 0] = mpx
            else:
                outdata[:, 0] = mpx
                outdata[:, 1] = mpx
            idx = end
            if idx >= stereo.shape[0]:
                idx = 0  # loop audio

        with sd.OutputStream(channels=1, dtype='float32', callback=callback, blocksize=blocksize):
            while not _stop_flag.is_set():
                time.sleep(0.25)

    except Exception as e:
        with _state_lock:
            STATE['last_error'] = str(e)
        log(f"Error: {e}")
    finally:
        with _state_lock:
            STATE['running'] = False
        _stop_flag.clear()
        log("Stream stopped")


@app.route('/')
def index():
    return render_template_string(TEMPLATE, devices=list_output_devices())


@app.post('/start')
def start():
    global _stream_thread
    if _stream_thread and _stream_thread.is_alive():
        log("Start requested, but stream already running")
        return redirect(url_for('index'))

    fs = int(request.form.get('fs', '192000'))
    device = request.form.get('device')
    device = int(device) if device else None
    source = request.form.get('source', 'tone')

    workdir = os.path.join(app.root_path, 'uploads')
    os.makedirs(workdir, exist_ok=True)

    audio_path = None
    tone = None
    duration = float(request.form.get('duration', '3600'))
    if source == 'file':
        f = request.files.get('audio')
        if f and f.filename:
            p = os.path.join(workdir, f"audio_{int(time.time())}")
            f.save(p)
            audio_path = p
    else:
        tone = float(request.form.get('tone', '1000'))

    pi_hex = request.form.get('pi', '0x1234')
    ps = request.form.get('ps', 'RADIO')
    rt = request.form.get('rt', 'Welcome')
    pilot = float(request.form.get('pilot', '0.08'))
    rds = float(request.form.get('rds', '0.03'))
    rds2 = float(request.form.get('rds2', '0.01'))
    enable_rds2 = request.form.get('enable_rds2') == 'on'

    logo_path = None
    lf = request.files.get('logo')
    if lf and lf.filename:
        p = os.path.join(workdir, f"logo_{int(time.time())}")
        lf.save(p)
        logo_path = p

    _stop_flag.clear()
    _stream_thread = threading.Thread(target=run_stream, kwargs=dict(
        fs=fs, device=device, audio_path=audio_path, tone=tone, duration=duration,
        pi_hex=pi_hex, ps=ps, rt=rt, pilot_level=pilot, rds_level=rds, rds2_level=rds2,
        enable_rds2=enable_rds2, logo_path=logo_path
    ), daemon=True)
    _stream_thread.start()

    return redirect(url_for('index'))


@app.route('/stop', methods=['POST', 'GET'])
def stop():
    _stop_flag.set()
    return (jsonify({'ok': True}), 200) if request.method == 'POST' else redirect(url_for('index'))


@app.get('/status')
def status():
    with _state_lock:
        running = STATE['running']
        started = STATE['started_at']
        uptime = int(time.time() - started) if running and started else 0
        resp = {
            'running': running,
            'device': STATE['device'],
            'fs': STATE['fs'],
            'source': STATE['source'],
            'uptime_s': uptime,
            'last_error': STATE['last_error'],
        }
    return jsonify(resp)


@app.get('/events')
def events():
    def gen():
        last_idx = 0
        while True:
            # Send status heartbeat every second
            with _state_lock:
                running = STATE['running']
                started = STATE['started_at']
                uptime = int(time.time() - started) if running and started else 0
                stat = {
                    'type': 'status',
                    'running': running,
                    'device': STATE['device'],
                    'fs': STATE['fs'],
                    'source': STATE['source'],
                    'uptime_s': uptime,
                    'last_error': STATE['last_error'],
                }
            yield f"data: {json.dumps(stat)}\n\n"

            # Flush new logs
            while last_idx < len(LOGS):
                line = list(LOGS)[last_idx]
                last_idx += 1
                yield f"data: {json.dumps({'type':'log','text':line})}\n\n"

            # Wait for new logs or timeout
            LOG_EVENT.clear()
            LOG_EVENT.wait(timeout=1.0)
    return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    # For production, run: waitress-serve --port=8080 webui:app
    app.run(host='0.0.0.0', port=8080, debug=True)