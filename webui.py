#!/usr/bin/env python3
import io
import os
import threading
import time
from typing import Optional

from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, jsonify
import numpy as np
import sounddevice as sd
import soundfile as sf

from rds2_stream import (
    RdsConfig,
    RdsBitstreamGenerator,
    make_mpx,
    read_audio_file,
    generate_tone,
    db_to_linear,
    RDS_BITRATE,
)

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
  <title>FM MPX + RDS/RDS2</title>
</head>
<body class="bg-slate-50 text-slate-800">
  <div class="max-w-5xl mx-auto p-6">
    <h1 class="text-2xl font-bold mb-4">FM MPX + RDS/RDS2 Web UI</h1>

    <form class="grid grid-cols-1 md:grid-cols-2 gap-4" method="post" enctype="multipart/form-data" action="{{ url_for('start') }}">
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
          <input class="border rounded px-2 py-1 w-full" type="number" step="1" name="duration" value="60" />
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
            <input type="checkbox" name="enable_rds2" checked /> Enable RDS2
          </label>
        </div>
        <div>
          <label class="block text-sm">Upload Logo (png/jpg)</label>
          <input class="border rounded px-2 py-1 w-full" type="file" name="logo" accept="image/*" />
        </div>
      </div>

      <div class="md:col-span-2 flex items-center gap-3">
        <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded" type="submit">Start</button>
        <a class="bg-gray-200 hover:bg-gray-300 px-4 py-2 rounded" href="{{ url_for('stop') }}">Stop</a>
        <span id="status" class="ml-2 text-sm"></span>
      </div>
    </form>
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
  </script>
</body>
</html>
"""

app = Flask(__name__)

_stream_thread: Optional[threading.Thread] = None
_stop_flag = threading.Event()


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
    sd.default.samplerate = fs
    if device is not None:
        sd.default.device = device

    if audio_path:
        stereo, _ = read_audio_file(audio_path, target_fs=fs)
    else:
        stereo = generate_tone(duration_s=duration, fs=fs, freq_hz=tone or 1000.0)

    cfg = RdsConfig(pi_code=int(pi_hex, 16), program_service_name=ps or '', radiotext=rt or '')
    gen = RdsBitstreamGenerator(cfg)
    if enable_rds2 and logo_path:
        from rds2_stream import load_logo_bits
        gen.set_logo_bits(load_logo_bits(logo_path))

    total_seconds = stereo.shape[0] / fs
    rds_bits = gen.generate_bits(int(total_seconds * RDS_BITRATE * 1.1))

    blocksize = 4096
    idx = 0
    gain = 1.0

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
            idx = 0  # loop audio for continuous streaming

    with sd.OutputStream(channels=1, dtype='float32', callback=callback, blocksize=blocksize):
        while not _stop_flag.is_set():
            time.sleep(0.1)


@app.route('/')
def index():
    return render_template_string(TEMPLATE, devices=list_output_devices())


@app.post('/start')
def start():
    global _stream_thread
    if _stream_thread and _stream_thread.is_alive():
        return redirect(url_for('index'))

    fs = int(request.form.get('fs', '192000'))
    device = request.form.get('device')
    device = int(device) if device else None
    source = request.form.get('source', 'tone')

    # Store uploads in tmp folder
    workdir = os.path.join(app.root_path, 'uploads')
    os.makedirs(workdir, exist_ok=True)

    audio_path = None
    tone = None
    duration = float(request.form.get('duration', '60'))
    if source == 'file':
        f = request.files.get('audio')
        if f and f.filename:
            p = os.path.join(workdir, 'audio_' + str(int(time.time())))
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
        p = os.path.join(workdir, 'logo_' + str(int(time.time())))
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


@app.get('/stop')
def stop():
    _stop_flag.set()
    return redirect(url_for('index'))


if __name__ == '__main__':
    # For production, run: waitress-serve --port=8080 webui:app
    app.run(host='0.0.0.0', port=8080, debug=True)