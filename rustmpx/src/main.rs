use clap::{Parser, Subcommand};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};

// Web + async
#[cfg(feature = "web")]
use axum::{routing::{get, post}, Router, extract::{Multipart, State}, response::{Html, Redirect}, Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

// Audio
#[cfg(feature = "audio")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "audio")]
use ringbuf::{HeapRb, Rb, Consumer, Producer};

// Persistence
#[cfg(feature = "web")]
use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};
use sqlx::Row;

// Files & images
use std::path::{Path, PathBuf};
use std::fs;
use image::{DynamicImage, GenericImageView, imageops::FilterType};

// Symphonia buffer helpers
use symphonia::core::audio::{AudioBufferRef, SampleBuffer, Signal};

#[derive(Parser)]
#[command(name = "rustmpx")] 
#[command(about = "FM MPX generator with RDS/RDS2 (Rust)", long_about = None)]
struct Cli {
	#[command(subcommand)]
	command: Commands,
}

#[derive(Subcommand)]
enum Commands {
	/// Generate a tone and write MPX WAV
	ToFile {
		#[arg(long, default_value_t = 192000)]
		fs: u32,
		#[arg(long, default_value_t = 30.0)]
		duration: f64,
		#[arg(long, default_value_t = 1000.0)]
		tone: f64,
		#[arg(long, default_value = "mpx.wav")]
		output: String,
		#[arg(long, default_value_t = 0x1234)]
		pi: u16,
		#[arg(long, default_value = "RADIO")] 
		ps: String,
		#[arg(long, default_value = "Welcome to RADIO")] 
		rt: String,
		#[arg(long, default_value_t = -3.0)]
		level_mpx: f64,
		#[arg(long, default_value_t = 0.08)]
		pilot: f64,
		#[arg(long, default_value_t = 0.03)]
		rds: f64,
		#[arg(long, default_value_t = 0.01)]
		rds2: f64,
		#[arg(long)]
		enable_rds2: bool,
	},
	/// List audio output devices
	Devices {
		#[arg(long, default_value_t = 192000)]
		fs: u32,
	},
	/// Play to an audio device (tone or file)
	Play {
		#[arg(long, default_value_t = 192000)]
		fs: u32,
		#[arg(long)]
		device_index: Option<usize>,
		#[arg(long)]
		input_file: Option<String>,
		#[arg(long, default_value_t = 1000.0)]
		tone: f64,
		#[arg(long, default_value_t = 0x1234)]
		pi: u16,
		#[arg(long, default_value = "RADIO")] 
		ps: String,
		#[arg(long, default_value = "Welcome to RADIO")] 
		rt: String,
		#[arg(long, default_value_t = 0.08)]
		pilot: f64,
		#[arg(long, default_value_t = 0.03)]
		rds: f64,
		#[arg(long, default_value_t = 0.01)]
		rds2: f64,
		#[arg(long)]
		enable_rds2: bool,
	},
	/// Run the web UI
	Serve {
		#[arg(long, default_value_t = 8080)]
		port: u16,
		#[arg(long, default_value = "/workspace/rustmpx/uploads")] 
		upload_dir: String,
	},
}

fn db_to_linear(db: f64) -> f64 { 10f64.powf(db / 20.0) }

fn clamp(sample: f32) -> f32 {
	if sample > 0.999 { 0.999 } else if sample < -0.999 { -0.999 } else { sample }
}

fn generate_tone_stereo(fs: u32, duration: f64, freq: f64, level_db: f64) -> (Vec<f32>, Vec<f32>) {
	let n = (duration * fs as f64) as usize;
	let amp = db_to_linear(level_db) as f32;
	let mut left = Vec::with_capacity(n);
	let mut right = Vec::with_capacity(n);
	for i in 0..n {
		let t = i as f64 / fs as f64;
		let s = (2.0 * PI * freq * t).sin() as f32 * amp;
		left.push(s);
		right.push(s);
	}
	(left, right)
}

// Simple linear resampler to match target fs
fn linear_resample_mono(input: &[f32], in_fs: u32, out_fs: u32) -> Vec<f32> {
	if in_fs == out_fs { return input.to_vec(); }
	let ratio = out_fs as f64 / in_fs as f64;
	let out_len = (input.len() as f64 * ratio) as usize;
	let mut out = vec![0.0f32; out_len];
	for i in 0..out_len {
		let src_pos = i as f64 / ratio;
		let idx = src_pos.floor() as usize;
		let frac = (src_pos - idx as f64) as f32;
		if idx + 1 < input.len() {
			let a = input[idx];
			let b = input[idx + 1];
			out[i] = a + (b - a) * frac;
		} else {
			out[i] = input[input.len() - 1];
		}
	}
	out
}

fn linear_resample_stereo(input_l: &[f32], input_r: &[f32], in_fs: u32, out_fs: u32) -> (Vec<f32>, Vec<f32>) {
	(
		linear_resample_mono(input_l, in_fs, out_fs),
		linear_resample_mono(input_r, in_fs, out_fs),
	)
}

// RDS basics
const RDS_BITRATE: f64 = 1187.5;
const PILOT_HZ: f64 = 19000.0;
const STEREO_SUB_HZ: f64 = 38000.0;
const RDS0_HZ: f64 = 57000.0;
const RDS2_SUBS: [f64; 3] = [66500.0, 76000.0, 85500.0];

// RDS2 experimental logo framing
const RDS2_LOGO_MAX_W: u32 = 64;
const RDS2_LOGO_MAX_H: u32 = 32;
const RDS2_LOGO_MAGIC: u8 = 0xA7;

// Minimal 0A + 2A generator (simplified), focusing on bitstream length and cadence
#[derive(Clone)]
struct RdsConfig { pi: u16, ps: String, rt: String }

fn rds_crc10(word: u16) -> u16 {
	let poly: u16 = 0x5B9; // as in Python
	let mut reg: u32 = 0;
	let data: u32 = (word as u32) << 10;
	let mut mask: u32 = 1 << 25;
	for _ in 0..26 {
		let bit = if (data & mask) != 0 { 1 } else { 0 };
		let top = if (reg & (1 << 10)) != 0 { 1 } else { 0 };
		reg = ((reg << 1) & 0x7FF) | bit as u32;
		if top == 1 { reg ^= poly as u32; }
		mask >>= 1;
	}
	(reg & 0x3FF) as u16
}

fn pack_block(word: u16, offset: u16) -> Vec<u8> {
	let cw = rds_crc10(word) ^ offset;
	let mut bits = Vec::with_capacity(26);
	for i in (0..16).rev() { bits.push(((word >> i) & 1) as u8); }
	for i in (0..10).rev() { bits.push(((cw >> i) & 1) as u8); }
	bits
}

fn group_0a(cfg: &RdsConfig, seg: usize) -> Vec<u8> {
	let ps = {
		let mut s = cfg.ps.clone();
		s.truncate(8);
		while s.len() < 8 { s.push(' '); }
		s
	};
	let c1 = ps.as_bytes()[((seg & 3) * 2) + 0] as u16;
	let c2 = ps.as_bytes()[((seg & 3) * 2) + 1] as u16;
	let block_a = cfg.pi;
	let mut block_b: u16 = 0; // type=0, version A(0)
	let group_type: u16 = 0;
	let version_a: u16 = 0;
	block_b |= (group_type & 0xF) << 1;
	block_b |= version_a & 0x1;
	block_b |= (seg as u16) & 0x3; // segment address
	let block_c: u16 = 0;
	let block_d: u16 = ((c1 & 0xFF) << 8) | (c2 & 0xFF);
	let mut out = Vec::new();
	out.extend(pack_block(block_a, 0x0FC));
	out.extend(pack_block(block_b, 0x198));
	out.extend(pack_block(block_c, 0x168));
	out.extend(pack_block(block_d, 0x1B4));
	out
}

fn group_2a(cfg: &RdsConfig, seg: usize) -> Vec<u8> {
	let mut text = cfg.rt.clone();
	text.truncate(64);
	while text.len() < 64 { text.push(' '); }
	let idx = seg & 0x0F;
	let b = text.as_bytes();
	let c1 = b[idx * 4 + 0] as u16;
	let c2 = b[idx * 4 + 1] as u16;
	let c3 = b[idx * 4 + 2] as u16;
	let c4 = b[idx * 4 + 3] as u16;
	let block_a = cfg.pi;
	let group_type: u16 = 2;
	let version_a: u16 = 0;
	let mut block_b: u16 = 0;
	block_b |= (group_type & 0xF) << 1;
	block_b |= version_a & 0x1;
	block_b |= (idx as u16) & 0x0F;
	let block_c: u16 = ((c1 & 0xFF) << 8) | (c2 & 0xFF);
	let block_d: u16 = ((c3 & 0xFF) << 8) | (c4 & 0xFF);
	let mut out = Vec::new();
	out.extend(pack_block(block_a, 0x0FC));
	out.extend(pack_block(block_b, 0x198));
	out.extend(pack_block(block_c, 0x168));
	out.extend(pack_block(block_d, 0x1B4));
	out
}

struct RdsGen {
	cfg: RdsConfig,
	ps_idx: usize,
	rt_idx: usize,
	logo_bits: Option<Vec<u8>>,
	logo_pos: usize,
}
impl RdsGen {
	fn new(cfg: RdsConfig) -> Self { Self { cfg, ps_idx: 0, rt_idx: 0, logo_bits: None, logo_pos: 0 } }
	fn set_logo_bits(&mut self, bits: Option<Vec<u8>>) { self.logo_bits = bits; self.logo_pos = 0; }
	fn next_logo_chunk(&mut self, max_bits: usize) -> Option<Vec<u8>> {
		if let Some(data) = &self.logo_bits {
			if data.is_empty() { return None; }
			let mut n = max_bits.min(data.len() - self.logo_pos);
			if n == 0 { self.logo_pos = 0; n = max_bits.min(data.len()); }
			let chunk = data[self.logo_pos..self.logo_pos + n].to_vec();
			self.logo_pos += n;
			return Some(chunk);
		}
		None
	}
	fn next_bits(&mut self) -> Vec<u8> {
		if (self.ps_idx % 5) == 0 {
			if let Some(ch) = self.next_logo_chunk(104) { return ch; }
		}
		if (self.ps_idx % 3) != 2 {
			let bits = group_0a(&self.cfg, self.ps_idx & 3);
			self.ps_idx = (self.ps_idx + 1) % 4;
			bits
		} else {
			let bits = group_2a(&self.cfg, self.rt_idx & 0x0F);
			self.rt_idx = (self.rt_idx + 1) % 16;
			bits
		}
	}
	fn generate(&mut self, total_bits: usize) -> Vec<u8> {
		let mut out = Vec::with_capacity(total_bits);
		while out.len() < total_bits {
			let g = self.next_bits();
			let n = std::cmp::min(g.len(), total_bits - out.len());
			out.extend_from_slice(&g[..n]);
		}
		out
	}
}

fn sinc(x: f64) -> f64 { if x == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) } }

fn raised_cosine(num_taps: usize, beta: f64, sps: f64) -> Vec<f64> {
	let mut h = vec![0.0; num_taps];
	let center = (num_taps as f64 - 1.0) / 2.0;
	for i in 0..num_taps {
		let t = (i as f64 - center) / sps;
		let denom = 1.0 - (2.0 * beta * t).powi(2);
		if denom.abs() < 1e-8 {
			h[i] = PI / 4.0 * sinc(1.0 / (2.0 * beta));
		} else {
			h[i] = sinc(t) * ((PI * beta * t).cos() / denom);
		}
	}
	let sum: f64 = h.iter().sum();
	for v in &mut h.iter_mut() { *v = *v / sum; }
	h
}

fn bpsk_subcarrier(bits: &[u8], fs: u32, sub_hz: f64, bitrate: f64) -> Vec<f32> {
	let sps = fs as f64 / bitrate;
	if sps < 4.0 { panic!("Sampling rate too low for RDS/RDS2"); }
	let symbols = {
		let mut out = Vec::with_capacity(bits.len());
		let mut phase = 1.0f64;
		for &b in bits { if b != 0 { phase = -phase; } out.push(phase); }
		out
	};
	let up = sps.round() as usize;
	let sps_eff = up as f64;
	let mut base = vec![0.0f64; symbols.len() * up];
	for (i, &sym) in symbols.iter().enumerate() { base[i * up] = sym; }
	let num_taps = std::cmp::max(41, ((6.0 * sps_eff) as usize) | 1);
	let h = raised_cosine(num_taps, 0.5, sps_eff);
	let mut shaped = vec![0.0f64; base.len()];
	let half = num_taps / 2;
	for n in 0..base.len() {
		let mut acc = 0.0;
		let start = if n >= half { n - half } else { 0 };
		let end = std::cmp::min(n + half, base.len() - 1);
		for i in start..=end {
			let k = i as isize - n as isize + half as isize;
			let h_idx = k as usize;
			acc += base[i] * h[h_idx];
		}
		shaped[n] = acc;
	}
	let n = shaped.len();
	let mut out = Vec::with_capacity(n);
	for i in 0..n {
		let t = i as f64 / fs as f64;
		let c = (2.0 * PI * sub_hz * t).cos();
		out.push((shaped[i] * c) as f32);
	}
	out
}

fn make_mpx(left: &[f32], right: &[f32], fs: u32, pilot: f64, rds: f64, rds2: f64, rds_bits: &[u8], enable_rds2: bool) -> Vec<f32> {
	assert_eq!(left.len(), right.len());
	let n = left.len();
	let mut out = Vec::with_capacity(n);
	let mut lpr = vec![0.0f32; n];
	let mut lmr = vec![0.0f32; n];
	for i in 0..n { lpr[i] = 0.5 * (left[i] + right[i]); lmr[i] = left[i] - right[i]; }
	let mut pilot_wave = vec![0.0f32; n];
	let mut dsb = vec![0.0f32; n];
	for i in 0..n {
		let t = i as f64 / fs as f64;
		pilot_wave[i] = (pilot * (2.0 * PI * PILOT_HZ * t).sin()) as f32;
		dsb[i] = (lmr[i] as f64 * (2.0 * PI * STEREO_SUB_HZ * t).cos()) as f32;
	}
	let rds_wave = bpsk_subcarrier(rds_bits, fs, RDS0_HZ, RDS_BITRATE);
	let mut rds_sum = vec![0.0f32; n];
	for i in 0..n.min(rds_wave.len()) { rds_sum[i] += (rds as f32) * rds_wave[i]; }
	if enable_rds2 {
		for &sc in &RDS2_SUBS {
			let r2 = bpsk_subcarrier(rds_bits, fs, sc, RDS_BITRATE);
			for i in 0..n.min(r2.len()) { rds_sum[i] += (rds2 as f32) * r2[i]; }
		}
	}
	for i in 0..n { out.push(clamp(lpr[i] + pilot_wave[i] + dsb[i] + rds_sum[i])); }
	out
}

fn write_wav_mono(path: &str, fs: u32, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
	let spec = hound::WavSpec { channels: 1, sample_rate: fs, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
	let mut writer = hound::WavWriter::create(path, spec)?;
	for &s in data {
		let v = (s.max(-1.0).min(1.0) * i16::MAX as f32) as i16;
		writer.write_sample(v)?;
	}
	writer.finalize()?;
	Ok(())
}

// ============ Audio device helpers ============
#[derive(Serialize)]
struct OutputDeviceInfo { index: usize, name: String, channels: u16 }

#[cfg(feature = "audio")]
fn list_output_devices() -> Vec<OutputDeviceInfo> {
	let host = cpal::default_host();
	let mut out = Vec::new();
	if let Ok(devs) = host.output_devices() {
		for (idx, d) in devs.enumerate() {
			let name = d.name().unwrap_or_else(|_| "Unknown".to_string());
			if let Ok(cfgs) = d.supported_output_configs() {
				let mut max_channels = 0u16;
				for cfg in cfgs { max_channels = max_channels.max(cfg.channels()); }
				if max_channels > 0 { out.push(OutputDeviceInfo { index: idx, name, channels: max_channels }); }
			}
		}
	}
	out
}

#[cfg(not(feature = "audio"))]
fn list_output_devices() -> Vec<OutputDeviceInfo> { vec![] }

// Shared streaming state for CLI and web
#[derive(Clone)]
struct StreamConfig {
	fs: u32,
	device_index: Option<usize>,
	source: SourceKind,
	pi: u16,
	ps: String,
	rt: String,
	pilot: f64,
	rds: f64,
	rds2: f64,
	enable_rds2: bool,
	logo_bits: Option<Vec<u8>>,
}
#[derive(Clone)]
enum SourceKind { Tone { freq: f64 }, File { path: String } }

struct RuntimeState {
	stop_flag: Arc<AtomicBool>,
	bg_task: Option<JoinHandle<()>>,
	current_cfg: Option<StreamConfig>,
	started_at: Option<std::time::Instant>,
}

impl RuntimeState {
	fn new() -> Self { Self { stop_flag: Arc::new(AtomicBool::new(false)), bg_task: None, current_cfg: None, started_at: None } }
}

fn decode_audio_file(path: &str) -> anyhow::Result<(Vec<f32>, Vec<f32>, u32)> {
	use symphonia::core::codecs::DecoderOptions;
	use symphonia::core::formats::FormatOptions;
	use symphonia::core::io::MediaSourceStream;
	use symphonia::core::meta::MetadataOptions;
	use symphonia::core::probe::Hint;

	let file = std::fs::File::open(path)?;
	let mss = MediaSourceStream::new(Box::new(file), Default::default());
	let mut hint = Hint::new();
	if let Some(ext) = Path::new(path).extension().and_then(|s| s.to_str()) { hint.with_extension(ext); }
	let probed = symphonia::default::get_probe().format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())?;
	let mut format = probed.format;
	let track = format.default_track().ok_or_else(|| anyhow::anyhow!("no default track"))?;
	let dec_opt = DecoderOptions { verify: true, ..Default::default() };
	let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opt)?;
	let src_fs = track.codec_params.sample_rate.ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
	let mut left = Vec::new();
	let mut right = Vec::new();
	loop {
		let packet = match format.next_packet() { Ok(p) => p, Err(_) => break };
		let audio_buf = decoder.decode(&packet)?;
		let spec = *audio_buf.spec();
		let chans = spec.channels.count();
		let mut tmp_l = Vec::new();
		let mut tmp_r = Vec::new();
		match audio_buf { 
			symphonia::core::audio::AudioBufferRef::F32(buf) => {
				let frames = buf.frames();
				let mut sbuf = SampleBuffer::<f32>::new(frames as u64, *buf.spec());
				sbuf.copy_interleaved_ref(symphonia::core::audio::AudioBufferRef::F32(buf));
				let data = sbuf.samples();
				for f in 0..frames {
					let l = data[f * chans];
					let r = if chans > 1 { data[f * chans + 1] } else { l };
					tmp_l.push(l);
					tmp_r.push(r);
				}
			}
			_ => {
				return Err(anyhow::anyhow!("only f32 audio source supported in this build"));
			}
		}
		left.extend(tmp_l);
		right.extend(tmp_r);
	}
	Ok((left, right, src_fs))
}

fn process_logo_to_bits(path: &str) -> anyhow::Result<Vec<u8>> {
	let img = image::open(path)?;
	let (w, h) = img.dimensions();
	let mut canvas = DynamicImage::new_luma8(RDS2_LOGO_MAX_W, RDS2_LOGO_MAX_H).to_luma8();
	let scale = f64::min(RDS2_LOGO_MAX_W as f64 / w as f64, RDS2_LOGO_MAX_H as f64 / h as f64);
	let new_w = (w as f64 * scale).round().max(1.0) as u32;
	let new_h = (h as f64 * scale).round().max(1.0) as u32;
	let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3).to_luma8();
	let off_x = ((RDS2_LOGO_MAX_W - new_w) / 2) as u32;
	let off_y = ((RDS2_LOGO_MAX_H - new_h) / 2) as u32;
	for y in 0..new_h {
		for x in 0..new_w {
			let p = *resized.get_pixel(x, y);
			canvas.put_pixel(off_x + x, off_y + y, p);
		}
	}
	// Adaptive threshold via mean
	let mean: f32 = canvas.pixels().map(|p| p.0[0] as f32).sum::<f32>() / (RDS2_LOGO_MAX_W * RDS2_LOGO_MAX_H) as f32;
	let mut bits = Vec::with_capacity((RDS2_LOGO_MAX_W * RDS2_LOGO_MAX_H) as usize + 16);
	bits.push(1); bits.push(0); // simple sync
	bits.push(RDS2_LOGO_MAGIC >> 7 & 1); bits.push(RDS2_LOGO_MAGIC >> 6 & 1); bits.push(RDS2_LOGO_MAGIC >> 5 & 1); bits.push(RDS2_LOGO_MAGIC >> 4 & 1);
	bits.push(RDS2_LOGO_MAGIC >> 3 & 1); bits.push(RDS2_LOGO_MAGIC >> 2 & 1); bits.push(RDS2_LOGO_MAGIC >> 1 & 1); bits.push(RDS2_LOGO_MAGIC & 1);
	for y in 0..RDS2_LOGO_MAX_H {
		for x in 0..RDS2_LOGO_MAX_W {
			let v = canvas.get_pixel(x, y).0[0] as f32;
			bits.push(if v > mean { 1 } else { 0 });
		}
	}
	Ok(bits)
}

#[cfg(feature = "audio")]
async fn start_stream(cfg: StreamConfig, state: Arc<Mutex<RuntimeState>>) -> anyhow::Result<()> {
	// Prepare audio source buffer aligned to fs
	let (mut left, mut right) = match cfg.source.clone() {
		SourceKind::Tone { freq } => {
			let (l, r) = generate_tone_stereo(cfg.fs, 60.0, freq, -6.0);
			(l, r)
		},
		SourceKind::File { path } => {
			let (l, r, src_fs) = decode_audio_file(&path)?;
			let (l2, r2) = linear_resample_stereo(&l, &r, src_fs, cfg.fs);
			(l2, r2)
		}
	};
	if left.is_empty() { left = vec![0.0; (cfg.fs as f64) as usize]; right = left.clone(); }
	let mut rds_gen = RdsGen::new(RdsConfig { pi: cfg.pi, ps: cfg.ps.clone(), rt: cfg.rt.clone() });
	rds_gen.set_logo_bits(cfg.logo_bits.clone());
	let mut rds_bits = rds_gen.generate((RDS_BITRATE * 2.0) as usize);

	let host = cpal::default_host();
	let device = if let Some(idx) = cfg.device_index {
		let mut it = host.output_devices()?;
		it.nth(idx).ok_or_else(|| anyhow::anyhow!("device index not found"))?
	} else { host.default_output_device().ok_or_else(|| anyhow::anyhow!("no default output device"))? };
	let mut supported = device.supported_output_configs()?;
	let mut chosen = None;
	while let Some(cfg) = supported.next() {
		let sr = cfg.min_sample_rate().0..=cfg.max_sample_rate().0;
		if sr.contains(&cfg.fs) || sr.contains(&cfg.fs) {} // dummy to satisfy borrow
	}
	let mut best_diff = u32::MAX;
	let mut best_cfg = None;
	for cfg in device.supported_output_configs()? {
		let range = cfg.min_sample_rate().0..=cfg.max_sample_rate().0;
		let target = if range.contains(&cfg.fs) { cfg.fs } else { cfg.min_sample_rate().0 };
		let diff = (target as i64 - cfg.fs as i64).unsigned_abs();
		if diff < best_diff { best_diff = diff; best_cfg = Some(cfg.with_sample_rate(cpal::SampleRate(target))); }
	}
	let out_cfg = best_cfg.ok_or_else(|| anyhow::anyhow!("no supported output config"))?;
	let sr = out_cfg.sample_rate().0;
	let channels = out_cfg.channels() as usize;

	let rb = HeapRb::<f32>::new((sr as usize) * 2);
	let (mut prod, mut cons) = rb.split();
	let stop = state.lock().unwrap().stop_flag.clone();

	let fs_target = cfg.fs;
	let enable_rds2 = cfg.enable_rds2;
	let pilot = cfg.pilot; let rds = cfg.rds; let rds2 = cfg.rds2;
	let mut pos = 0usize;
	let bg = tokio::spawn(async move {
		let mut local_left = left;
		let mut local_right = right;
		loop {
			if stop.load(Ordering::SeqCst) { break; }
			let need = prod.free_len().min(1024);
			if need == 0 { tokio::time::sleep(Duration::from_millis(5)).await; continue; }
			let samples = need;
			let end = pos + samples;
			if end > local_left.len() { pos = 0; }
			let l = &local_left[pos..pos + samples.min(local_left.len()-pos)];
			let r = &local_right[pos..pos + samples.min(local_right.len()-pos)];
			let bits_needed = ((samples as f64 / fs_target as f64) * RDS_BITRATE) as usize + 208;
			if rds_bits.len() < bits_needed { rds_bits.extend(rds_gen.generate((RDS_BITRATE * 2.0) as usize)); }
			let bits_block: Vec<u8> = rds_bits.drain(0..bits_needed.min(rds_bits.len())).collect();
			let mpx = make_mpx(l, r, fs_target, pilot, rds, rds2, &bits_block, enable_rds2);
			for &s in &mpx { let _ = prod.push(s); }
			pos += samples;
		}
	});

	let err_fn = |e| eprintln!("stream error: {}", e);
	let stream = match out_cfg.sample_format() {
		cpal::SampleFormat::F32 => device.build_output_stream(&out_cfg.config(), move |data: &mut [f32], _| write_from_rb(data, channels, &mut cons), err_fn, None)?,
		cpal::SampleFormat::I16 => device.build_output_stream(&out_cfg.config(), move |data: &mut [i16], _| write_from_rb_i16(data, channels, &mut cons), err_fn, None)?,
		cpal::SampleFormat::U16 => device.build_output_stream(&out_cfg.config(), move |data: &mut [u16], _| write_from_rb_u16(data, channels, &mut cons), err_fn, None)?,
		_ => anyhow::bail!("unsupported sample format"),
	};
	stream.play()?;
	state.lock().unwrap().bg_task = Some(bg);
	Ok(())
}

#[cfg(not(feature = "audio"))]
async fn start_stream(_cfg: StreamConfig, _state: Arc<Mutex<RuntimeState>>) -> anyhow::Result<()> { Ok(()) }

#[cfg(feature = "audio")]
fn write_from_rb(out: &mut [f32], channels: usize, cons: &mut Consumer<f32>) {
	let frames = out.len() / channels;
	for f in 0..frames {
		let v = cons.pop().unwrap_or(0.0);
		for c in 0..channels { out[f * channels + c] = v; }
	}
}
#[cfg(feature = "audio")]
fn write_from_rb_i16(out: &mut [i16], channels: usize, cons: &mut Consumer<f32>) {
	let frames = out.len() / channels;
	for f in 0..frames {
		let v = cons.pop().unwrap_or(0.0);
		let s = (v.max(-1.0).min(1.0) * i16::MAX as f32) as i16;
		for c in 0..channels { out[f * channels + c] = s; }
	}
}
#[cfg(feature = "audio")]
fn write_from_rb_u16(out: &mut [u16], channels: usize, cons: &mut Consumer<f32>) {
	let frames = out.len() / channels;
	for f in 0..frames {
		let v = cons.pop().unwrap_or(0.0);
		let s = ((v.max(-1.0).min(1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16;
		for c in 0..channels { out[f * channels + c] = s; }
	}
}

async fn stop_stream(state: Arc<Mutex<RuntimeState>>) {
	let mut s = state.lock().unwrap();
	s.stop_flag.store(true, Ordering::SeqCst);
	if let Some(h) = s.bg_task.take() { let _ = h.abort(); }
}

// ============ Web UI ============
#[cfg(feature = "web")]
static TEMPLATE: &str = r#"<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <script src=\"https://cdn.tailwindcss.com\"></script>
  <title>rustmpx — FM MPX + RDS/RDS2</title>
  <style>
    :root { color-scheme: dark; }
    .card { background: linear-gradient(180deg, rgba(18, 18, 21, 0.9), rgba(14, 14, 18, 0.9)); border: 1px solid rgba(255,255,255,0.06); }
    .btn-primary { background: linear-gradient(90deg,#22d3ee,#4ade80); }
    .btn-primary:hover { filter: brightness(1.05); }
    .chip { border: 1px solid rgba(255,255,255,0.08); }
    .grid-form label { color: #9aa0a6; font-size: 0.85rem; }
    .dropzone { border: 1px dashed rgba(255,255,255,0.15); }
  </style>
</head>
<body class=\"bg-black text-white min-h-screen\">
  <div class=\"max-w-6xl mx-auto px-6 py-6\">
	<header class=\"flex items-center justify-between mb-6\">
	  <div class=\"flex items-center gap-3\">
		<div class=\"h-9 w-9 rounded bg-gradient-to-br from-cyan-400 to-emerald-400\"></div>
		<h1 class=\"text-xl font-semibold tracking-tight\">rustmpx</h1>
	  </div>
	  <div id=\"statusChip\" class=\"chip px-3 py-1 rounded text-sm text-gray-300\">Idle</div>
	</header>

	<div class=\"grid grid-cols-1 lg:grid-cols-3 gap-6\">
	  <section class=\"lg:col-span-2 card rounded-xl p-5\">
		<h2 class=\"text-lg font-medium mb-4\">Source & Output</h2>
		<form id=\"controlForm\" class=\"grid grid-cols-1 md:grid-cols-2 gap-4 grid-form\" enctype=\"multipart/form-data\">
		  <div>
			<label>Sample Rate (Hz)</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" type=\"number\" name=\"fs\" value=\"192000\" />
		  </div>
		  <div>
			<label>Output Device</label>
			<select id=\"deviceSelect\" class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" name=\"device\"></select>
		  </div>
		  <div class=\"md:col-span-2\">
			<div class=\"flex items-center gap-6\">
			  <label class=\"inline-flex items-center gap-2\"><input type=\"radio\" name=\"source\" value=\"tone\" checked /> Tone</label>
			  <label class=\"inline-flex items-center gap-2\"><input type=\"radio\" name=\"source\" value=\"file\" /> File</label>
			</div>
		  </div>
		  <div id=\"toneFields\">
			<label>Tone Frequency (Hz)</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" type=\"number\" name=\"tone\" value=\"1000\" />
			<label class=\"mt-3\">Duration (s)</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" type=\"number\" name=\"duration\" value=\"60\" />
		  </div>
		  <div id=\"fileFields\" class=\"hidden\">
			<label>Audio File</label>
			<div id=\"dropzone\" class=\"dropzone mt-1 rounded px-4 py-6 text-sm text-gray-300 flex items-center justify-center\">Drop audio here or click to browse</div>
			<input id=\"audioInput\" class=\"hidden\" type=\"file\" name=\"audio\" accept=\"audio/*\" />
		  </div>
		  <div class=\"md:col-span-2 flex items-center gap-3 mt-2\">
			<button id=\"startBtn\" class=\"btn-primary px-4 py-2 rounded text-black font-medium\" type=\"submit\">Start</button>
			<button id=\"stopBtn\" class=\"px-4 py-2 rounded border border-white/10\" type=\"button\">Stop</button>
		  </div>
		</form>
	  </section>

	  <section class=\"card rounded-xl p-5\">
		<h2 class=\"text-lg font-medium mb-4\">RDS / RDS2</h2>
		<form id=\"rdsForm\" class=\"grid grid-cols-2 gap-4 grid-form\">
		  <div>
			<label>PI (hex)</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" name=\"pi\" value=\"0x1234\" />
		  </div>
		  <div>
			<label>PS (name)</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" name=\"ps\" value=\"RADIO\" />
		  </div>
		  <div class=\"col-span-2\">
			<label>Radiotext</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" name=\"rt\" value=\"Welcome to RADIO\" />
		  </div>
		  <div>
			<label>Pilot</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" name=\"pilot\" value=\"0.08\" />
		  </div>
		  <div>
			<label>RDS</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" name=\"rds\" value=\"0.03\" />
		  </div>
		  <div>
			<label>RDS2</label>
			<input class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" name=\"rds2\" value=\"0.01\" />
		  </div>
		  <div class=\"col-span-2\">
			<label class=\"inline-flex items-center gap-2\"><input type=\"checkbox\" name=\"enable_rds2\" checked /> Enable RDS2</label>
		  </div>
		  <div class=\"col-span-2\">
			<label>Logo (auto-fit 64x32)</label>
			<input id=\"logoInput\" class=\"mt-1 w-full bg-transparent border rounded px-3 py-2 border-white/10\" type=\"file\" name=\"logo\" accept=\"image/*\" />
			<canvas id=\"logoPreview\" class=\"mt-2 rounded bg-white/5 w-full h-24\"></canvas>
		  </div>
		</form>
	  </section>
	</div>

	<section class=\"mt-6 card rounded-xl p-5\">
	  <h2 class=\"text-lg font-medium mb-3\">Activity</h2>
	  <pre id=\"log\" class=\"text-xs text-gray-300 whitespace-pre-wrap\"></pre>
	</section>
  </div>

  <script>
    async function fetchJSON(url){ const r = await fetch(url); if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); }
    async function refreshDevices(){ try { const list = await fetchJSON('/devices'); const sel = document.getElementById('deviceSelect'); sel.innerHTML = '<option value="">Default</option>'; list.forEach(d=>{ const o=document.createElement('option'); o.value=d.index; o.textContent=`${d.index} · ${d.name} (${d.channels}ch)`; sel.appendChild(o); }); } catch(e){ console.warn(e); } }
    async function refreshStatus(){ try { const st = await fetchJSON('/status'); const chip = document.getElementById('statusChip'); chip.textContent = st.streaming ? `Streaming · ${st.fs} Hz` : 'Idle'; chip.className = 'chip px-3 py-1 rounded text-sm ' + (st.streaming ? 'text-emerald-300' : 'text-gray-300'); } catch(e){} }
    async function refreshConfig(){ try { const c = await fetchJSON('/config'); if(!c) return; document.querySelector('input[name=\"fs\"]').value=c.fs||192000; document.querySelector('input[name=\"pi\"]').value=c.pi_hex||'0x1234'; document.querySelector('input[name=\"ps\"]').value=c.ps||''; document.querySelector('input[name=\"rt\"]').value=c.rt||''; document.querySelector('input[name=\"pilot\"]').value=c.pilot||0.08; document.querySelector('input[name=\"rds\"]').value=c.rds||0.03; document.querySelector('input[name=\"rds2\"]').value=c.rds2||0.01; if(c.enable_rds2){ document.querySelector('input[name=\"enable_rds2\"]').checked=true; } } catch(e){} }

    function hookSourceToggle(){ const radios = document.querySelectorAll('input[name=source]'); const tone = document.getElementById('toneFields'); const file = document.getElementById('fileFields'); const update = ()=>{ const v = document.querySelector('input[name=source]:checked').value; tone.classList.toggle('hidden', v!=='tone'); file.classList.toggle('hidden', v!=='file'); }; radios.forEach(r=>r.addEventListener('change', update)); update(); }

    function setupDropzone(){ const dz=document.getElementById('dropzone'); const inp=document.getElementById('audioInput'); dz.addEventListener('click',()=>inp.click()); dz.addEventListener('dragover',(e)=>{ e.preventDefault(); dz.classList.add('border-cyan-400');}); dz.addEventListener('dragleave',()=>dz.classList.remove('border-cyan-400')); dz.addEventListener('drop',(e)=>{ e.preventDefault(); dz.classList.remove('border-cyan-400'); if(e.dataTransfer.files.length){ inp.files=e.dataTransfer.files; dz.textContent=e.dataTransfer.files[0].name; } }); inp.addEventListener('change',()=>{ if(inp.files.length){ dz.textContent=inp.files[0].name; } }); }

    function setupLogoPreview(){ const inp=document.getElementById('logoInput'); const canvas=document.getElementById('logoPreview'); const ctx=canvas.getContext('2d'); canvas.width=512; canvas.height=96; inp.addEventListener('change',()=>{ const f=inp.files[0]; if(!f) return; const img=new Image(); img.onload=()=>{ const w=img.width,h=img.height; const scale=Math.min(canvas.width*0.9/w, canvas.height*0.9/h); const nw=w*scale, nh=h*scale; ctx.clearRect(0,0,canvas.width,canvas.height); ctx.globalAlpha=0.3; ctx.fillStyle='#0a0a0a'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.globalAlpha=1.0; ctx.drawImage(img,(canvas.width-nw)/2,(canvas.height-nh)/2,nw,nh); }; img.src=URL.createObjectURL(f); }); }

    async function startStreaming(){ const cf=document.getElementById('controlForm'); const rf=document.getElementById('rdsForm'); const fd=new FormData(cf); new FormData(rf).forEach((v,k)=>fd.append(k,v)); const r=await fetch('/start',{ method:'POST', body:fd }); if(r.redirected){ window.location=r.url; } }
    async function stopStreaming(){ await fetch('/stop'); }

    document.getElementById('controlForm').addEventListener('submit', async (e)=>{ e.preventDefault(); try{ await startStreaming(); }catch(err){ console.error(err);} finally{ setTimeout(refreshStatus,500); }});
    document.getElementById('stopBtn').addEventListener('click', async ()=>{ await stopStreaming(); setTimeout(refreshStatus,200); });

    hookSourceToggle(); setupDropzone(); setupLogoPreview();
    refreshDevices(); refreshConfig(); refreshStatus();
    setInterval(refreshStatus, 2000);
  </script>
</body>
</html>"#;

#[cfg(feature = "web")]
#[derive(Clone)]
struct AppContext { pool: SqlitePool, state: Arc<Mutex<RuntimeState>>, upload_dir: PathBuf }

#[cfg(feature = "web")]
async fn index_handler() -> Html<&'static str> { Html(TEMPLATE) }

#[cfg(feature = "web")]
async fn devices_handler() -> Json<Vec<OutputDeviceInfo>> { Json(list_output_devices()) }

#[cfg(feature = "web")]
#[derive(Serialize)]
struct Status { streaming: bool, fs: Option<u32>, since_ms: Option<u128> }

#[cfg(feature = "web")]
async fn status_handler(State(ctx): State<AppContext>) -> Json<Status> {
	let st = ctx.state.lock().unwrap();
	let streaming = st.bg_task.is_some();
	let fs = st.current_cfg.as_ref().map(|c| c.fs);
	let since_ms = st.started_at.map(|t| t.elapsed().as_millis());
	Json(Status { streaming, fs, since_ms })
}

#[cfg(feature = "web")]
#[derive(Serialize)]
struct SavedConfig { fs: u32, pi_hex: String, ps: String, rt: String, pilot: f64, rds: f64, rds2: f64, enable_rds2: bool }

#[cfg(feature = "web")]
async fn config_handler(State(ctx): State<AppContext>) -> Json<Option<SavedConfig>> {
	if let Ok(row) = sqlx::query("SELECT fs, pi, ps, rt, pilot, rds, rds2, enable_rds2 FROM config WHERE id=1").fetch_one(&ctx.pool).await {
		let fs: i64 = row.get::<i64, _>("fs");
		let pi: i64 = row.get::<i64, _>("pi");
		let ps: String = row.try_get::<String, _>("ps").unwrap_or_default();
		let rt: String = row.try_get::<String, _>("rt").unwrap_or_default();
		let pilot: f64 = row.try_get::<f64, _>("pilot").unwrap_or(0.08);
		let rds: f64 = row.try_get::<f64, _>("rds").unwrap_or(0.03);
		let rds2: f64 = row.try_get::<f64, _>("rds2").unwrap_or(0.01);
		let enable_rds2: i64 = row.try_get::<i64, _>("enable_rds2").unwrap_or(1);
		return Json(Some(SavedConfig { fs: fs as u32, pi_hex: format!("0x{:04X}", pi as u16), ps, rt, pilot, rds, rds2, enable_rds2: enable_rds2 != 0 }));
	}
	Json(None)
}

#[cfg(feature = "web")]
async fn start_handler(State(ctx): State<AppContext>, mut multipart: Multipart) -> Result<Redirect, String> {
	let upload_dir = ctx.upload_dir.clone();
	fs::create_dir_all(&upload_dir).map_err(|e| e.to_string())?;
	let mut fs_val: u32 = 192000;
	let mut source = "tone".to_string();
	let mut tone: f64 = 1000.0;
	let mut duration: f64 = 60.0;
	let mut device_index: Option<usize> = None;
	let mut audio_path: Option<PathBuf> = None;
	let mut pi: u16 = 0x1234;
	let mut ps = "RADIO".to_string();
	let mut rt = "Welcome to RADIO".to_string();
	let mut pilot = 0.08f64; let mut rds = 0.03f64; let mut rds2 = 0.01f64; let mut enable_rds2 = true;
	let mut logo_bits: Option<Vec<u8>> = None;
	while let Some(field) = multipart.next_field().await.map_err(|e| e.to_string())? {
		let name = field.name().unwrap_or("").to_string();
		if name == "audio" {
			if let Some(fname) = field.file_name().map(|s| s.to_string()) {
				let p = upload_dir.join(format!("audio_{}_{}", chrono::Utc::now().timestamp(), fname));
				let data = field.bytes().await.map_err(|e| e.to_string())?;
				fs::write(&p, &data).map_err(|e| e.to_string())?;
				audio_path = Some(p);
			}
		} else if name == "logo" {
			if let Some(fname) = field.file_name().map(|s| s.to_string()) {
				let p = upload_dir.join(format!("logo_{}_{}", chrono::Utc::now().timestamp(), fname));
				let data = field.bytes().await.map_err(|e| e.to_string())?;
				fs::write(&p, &data).map_err(|e| e.to_string())?;
				logo_bits = Some(process_logo_to_bits(p.to_str().unwrap()).map_err(|e| e.to_string())?);
			}
		} else if name == "fs" {
			fs_val = field.text().await.map_err(|e| e.to_string())?.parse().unwrap_or(192000);
		} else if name == "source" {
			source = field.text().await.map_err(|e| e.to_string())?;
		} else if name == "tone" {
			tone = field.text().await.map_err(|e| e.to_string())?.parse().unwrap_or(1000.0);
		} else if name == "duration" {
			duration = field.text().await.map_err(|e| e.to_string())?.parse().unwrap_or(60.0);
		} else if name == "device" {
			let v = field.text().await.map_err(|e| e.to_string())?;
			if !v.is_empty() { device_index = Some(v.parse().unwrap_or(0)); }
		} else if name == "pi" {
			let v = field.text().await.map_err(|e| e.to_string())?; pi = u16::from_str_radix(v.trim_start_matches("0x"), 16).unwrap_or(0x1234);
		} else if name == "ps" { ps = field.text().await.map_err(|e| e.to_string())?; }
		else if name == "rt" { rt = field.text().await.map_err(|e| e.to_string())?; }
		else if name == "pilot" { pilot = field.text().await.map_err(|e| e.to_string())?.parse().unwrap_or(0.08); }
		else if name == "rds" { rds = field.text().await.map_err(|e| e.to_string())?.parse().unwrap_or(0.03); }
		else if name == "rds2" { rds2 = field.text().await.map_err(|e| e.to_string())?.parse().unwrap_or(0.01); }
		else if name == "enable_rds2" { enable_rds2 = true; }
	}
	let src_kind = if source == "file" { SourceKind::File { path: audio_path.and_then(|p| p.to_str().map(|s| s.to_string())).ok_or_else(|| "audio file missing".to_string())? } } else { SourceKind::Tone { freq: tone } };
	let cfg = StreamConfig { fs: fs_val, device_index, source: src_kind, pi, ps: ps.clone(), rt: rt.clone(), pilot, rds, rds2, enable_rds2, logo_bits: logo_bits.clone() };
	// Persist
	let _ = sqlx::query("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, fs INTEGER, device_index INTEGER, pi INTEGER, ps TEXT, rt TEXT, pilot REAL, rds REAL, rds2 REAL, enable_rds2 INTEGER)").execute(&ctx.pool).await;
	let _ = sqlx::query("INSERT OR REPLACE INTO config (id, fs, device_index, pi, ps, rt, pilot, rds, rds2, enable_rds2) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
		.bind(fs_val as i64).bind(device_index.map(|v| v as i64)).bind(pi as i64).bind(ps).bind(rt).bind(pilot).bind(rds).bind(rds2).bind(if enable_rds2 {1} else {0})
		.execute(&ctx.pool).await;
	// Stop any existing stream then start
	stop_stream(ctx.state.clone()).await;
	{ let mut s = ctx.state.lock().unwrap(); s.stop_flag.store(false, Ordering::SeqCst); s.bg_task = None; s.current_cfg = Some(cfg.clone()); s.started_at = Some(std::time::Instant::now()); }
	start_stream(cfg, ctx.state.clone()).await.map_err(|e| e.to_string())?;
	Ok(Redirect::to("/"))
}

#[cfg(feature = "web")]
async fn stop_handler(ctx: axum::extract::State<AppContext>) -> Redirect { stop_stream(ctx.state.clone()).await; Redirect::to("/") }

#[cfg(feature = "web")]
async fn run_server(port: u16, upload_dir: PathBuf) -> anyhow::Result<()> {
	let db_path = upload_dir.join("db.sqlite");
	fs::create_dir_all(&upload_dir)?;
	let pool = SqlitePoolOptions::new().max_connections(5).connect(&format!("sqlite://{}", db_path.to_string_lossy())).await?;
	let state = Arc::new(Mutex::new(RuntimeState::new()));
	let ctx = AppContext { pool, state, upload_dir };
	let app = Router::new()
		.route("/", get(index_handler))
		.route("/devices", get(devices_handler))
		.route("/status", get(status_handler))
		.route("/config", get(config_handler))
		.route("/start", post(start_handler))
		.route("/stop", get(stop_handler))
		.with_state(ctx);
	let addr = std::net::SocketAddr::from(([0,0,0,0], port));
	axum::Server::bind(&addr).serve(app.into_make_service()).await?;
	Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let cli = Cli::parse();
	match cli.command {
		Commands::ToFile { fs, duration, tone, output, pi, ps, rt, level_mpx, pilot, rds, rds2, enable_rds2 } => {
			let (left, right) = generate_tone_stereo(fs, duration, tone, level_mpx);
			let mut gen = RdsGen::new(RdsConfig { pi, ps, rt });
			let bits_needed = (duration * RDS_BITRATE * 1.1) as usize;
			let rds_bits = gen.generate(bits_needed);
			let mpx = make_mpx(&left, &right, fs, pilot, rds, rds2, &rds_bits, enable_rds2);
			write_wav_mono(&output, fs, &mpx)?;
			println!("Wrote {} samples to {}", mpx.len(), output);
		}
		Commands::Devices { fs: _ } => {
			for d in list_output_devices() { println!("{} | {} | {}ch", d.index, d.name, d.channels); }
		}
		Commands::Play { fs, device_index, input_file, tone, pi, ps, rt, pilot, rds, rds2, enable_rds2 } => {
			let source = if let Some(p) = input_file { SourceKind::File { path: p } } else { SourceKind::Tone { freq: tone } };
			let cfg = StreamConfig { fs, device_index, source, pi, ps, rt, pilot, rds, rds2, enable_rds2, logo_bits: None };
			let state = Arc::new(Mutex::new(RuntimeState::new()));
			start_stream(cfg, state.clone()).await?;
			println!("Playing... Ctrl+C to stop");
			loop { tokio::time::sleep(Duration::from_secs(1)).await; }
		}
		Commands::Serve { port, upload_dir } => {
			#[cfg(feature = "web")]
			run_server(port, PathBuf::from(upload_dir)).await?;
			#[cfg(not(feature = "web"))]
			println!("Web feature not enabled in this build");
		}
	}
	Ok(())
}