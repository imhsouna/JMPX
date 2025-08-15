use clap::{Parser, Subcommand};
use std::f64::consts::PI;

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

// RDS basics
const RDS_BITRATE: f64 = 1187.5;
const PILOT_HZ: f64 = 19000.0;
const STEREO_SUB_HZ: f64 = 38000.0;
const RDS0_HZ: f64 = 57000.0;
const RDS2_SUBS: [f64; 3] = [66500.0, 76000.0, 85500.0];

// Minimal 0A + 2A generator (simplified), focusing on bitstream length and cadence
#[derive(Clone)]
struct RdsConfig { pi: u16, ps: String, rt: String }

fn rds_crc10(word: u16) -> u16 {
	let poly: u16 = 0x5B9; // as in Python
	let mut reg: u32 = 0;
	let mut data: u32 = (word as u32) << 10;
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
	block_b |= (version_a & 0x1);
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
	block_b |= (version_a & 0x1);
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

struct RdsGen { cfg: RdsConfig, ps_idx: usize, rt_idx: usize }
impl RdsGen {
	fn new(cfg: RdsConfig) -> Self { Self { cfg, ps_idx: 0, rt_idx: 0 } }
	fn next_bits(&mut self) -> Vec<u8> {
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

fn differential_encode(bits: &[u8]) -> Vec<f64> {
	let mut out = Vec::with_capacity(bits.len());
	let mut phase = 1.0f64;
	for &b in bits {
		if b != 0 { phase = -phase; }
		out.push(phase);
	}
	out
}

fn sinc(x: f64) -> f64 {
	if x == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) }
}

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
	let symbols = differential_encode(bits);
	let up = sps.round() as usize;
	let sps_eff = up as f64;
	let mut base = vec![0.0f64; symbols.len() * up];
	for (i, &sym) in symbols.iter().enumerate() { base[i * up] = sym; }
	let num_taps = std::cmp::max(41, ((6.0 * sps_eff) as usize) | 1);
	let h = raised_cosine(num_taps, 0.5, sps_eff);
	let mut shaped = vec![0.0f64; base.len()];
	// convolution same-length
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
	// No audio lowpass here for brevity; assume input is clean baseband
	let mut lpr = vec![0.0f32; n];
	let mut lmr = vec![0.0f32; n];
	for i in 0..n {
		lpr[i] = 0.5 * (left[i] + right[i]);
		lmr[i] = left[i] - right[i];
	}
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
	for i in 0..n {
		let v = lpr[i] + pilot_wave[i] + dsb[i] + rds_sum[i];
		out.push(clamp(v));
	}
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
	}
	Ok(())
}