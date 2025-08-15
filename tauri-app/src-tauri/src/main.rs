#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use tauri::State;

#[derive(Clone, Serialize, Deserialize)]
struct AppConfig {
	fs: u32,
	pi_hex: String,
	ps: String,
	rt: String,
	pilot: f64,
	rds: f64,
	rds2: f64,
	enable_rds2: bool,
}

#[derive(Default)]
struct Runtime {
	stop: Arc<AtomicBool>,
	is_streaming: bool,
}

#[tauri::command]
fn list_devices() -> Vec<String> {
	let host = cpal::default_host();
	let mut names = Vec::new();
	if let Ok(devs) = host.output_devices() {
		for d in devs { names.push(d.name().unwrap_or_else(|_| "Unknown".into())); }
	}
	names
}

#[tauri::command]
async fn start_stream(_cfg: AppConfig, runtime: State<'_, Arc<Mutex<Runtime>>>) -> Result<(), String> {
	let mut rt = runtime.lock().unwrap();
	rt.stop.store(false, Ordering::SeqCst);
	rt.is_streaming = true;
	Ok(())
}

#[tauri::command]
async fn stop_stream(runtime: State<'_, Arc<Mutex<Runtime>>>) -> Result<(), String> {
	let mut rt = runtime.lock().unwrap();
	rt.stop.store(true, Ordering::SeqCst);
	rt.is_streaming = false;
	Ok(())
}

#[tauri::command]
fn status(runtime: State<'_, Arc<Mutex<Runtime>>>) -> bool { runtime.lock().unwrap().is_streaming }

fn main() {
	let runtime: Arc<Mutex<Runtime>> = Arc::new(Mutex::new(Runtime::default()));
	tauri::Builder::default()
		.manage(runtime)
		.invoke_handler(tauri::generate_handler![list_devices, start_stream, stop_stream, status])
		.run(tauri::generate_context!())
		.expect("error while running tauri application");
}