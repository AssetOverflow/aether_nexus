//! Structured logging for AetherNexus.
//!
//! Provides two output channels:
//! - **File**: All messages (Trace and above) written to `logs/nexus-YYYY-MM-DD.log`
//! - **Console**: Only Info and above by default; Debug with `--verbose`
//!
//! # Usage
//!
//! ```rust
//! // Initialize early in main()
//! nexus_core::logging::init(false, false); // false = normal, true = verbose, false = no thinking
//!
//! // Then use the macros anywhere
//! use nexus_core::{nexus_info, nexus_debug, nexus_warn};
//! nexus_info!("Fabric mapped: {} MB", 1024);
//! nexus_debug!("KV cache elem_offset = {}", 42);
//! nexus_warn!("Weight embed failed: {}", "error");
//! ```

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{mpsc, OnceLock};
use std::thread;
use std::time::SystemTime;

// ─────────────────────────────────────────────────────────────────────────────
// Log Level
// ─────────────────────────────────────────────────────────────────────────────

/// Logging verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl Level {
    pub fn as_str(&self) -> &'static str {
        match self {
            Level::Trace => "TRACE",
            Level::Debug => "DEBUG",
            Level::Info  => "INFO ",
            Level::Warn  => "WARN ",
            Level::Error => "ERROR",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Logger
// ─────────────────────────────────────────────────────────────────────────────

struct LogMessage {
    level: Level,
    target: String,
    msg: String,
    timestamp: u64,
}

pub struct NexusLogger {
    /// Sender channel for fast, lock-free logging
    sender: mpsc::Sender<LogMessage>,
    /// Minimum level for console output.
    #[allow(dead_code)]
    console_level: Level,
    /// Whether to show full boot diagnostics on console.
    pub verbose: bool,
    /// Whether to show <think></think> reasoning blocks on console.
    pub show_thinking: bool,
}

impl NexusLogger {
    fn new(verbose: bool, show_thinking: bool) -> Self {
        let (tx, rx) = mpsc::channel::<LogMessage>();
        let console_level = if verbose { Level::Debug } else { Level::Info };

        // Spawn detached background thread for File I/O and Console writes
        thread::Builder::new()
            .name("nexus-logger".into())
            .spawn(move || {
                let mut file = Self::open_log_file().ok().map(BufWriter::new);
                
                // Keep draining until channel disconnects
                while let Ok(log) = rx.recv() {
                    let ts = log.timestamp;
                    let h = (ts % 86400) / 3600;
                    let m = (ts % 3600) / 60;
                    let s = ts % 60;

                    // Write to file
                    if let Some(ref mut f) = file {
                        let _ = writeln!(
                            f,
                            "{:02}:{:02}:{:02} {} [{}] {}",
                            h, m, s, log.level.as_str(), log.target, log.msg
                        );
                        if log.level >= Level::Warn {
                            let _ = f.flush();
                        }
                    }

                    // Write to console
                    if log.level >= console_level {
                        if log.level >= Level::Warn {
                            eprintln!("[{}] {}", log.level.as_str().trim(), log.msg);
                        } else if log.level == Level::Info {
                            println!("{}", log.msg);
                        } else if log.level == Level::Debug {
                            println!("  \x1b[2m{}\x1b[0m", log.msg);
                        }
                    }
                }
                
                // Flush on shutdown when channel closes
                if let Some(mut f) = file {
                    let _ = f.flush();
                }
            })
            .expect("Failed to spawn logger thread");

        Self {
            sender: tx,
            console_level,
            verbose,
            show_thinking,
        }
    }

    fn open_log_file() -> std::io::Result<fs::File> {
        fs::create_dir_all("logs")?;
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let days = now.as_secs() / 86400;
        let year = 1970 + days / 365;
        let remaining = days % 365;
        let month = (remaining / 30) + 1;
        let day = (remaining % 30) + 1;
        let path = PathBuf::from(format!("logs/nexus-{:04}-{:02}-{:02}.log", year, month, day));
        OpenOptions::new().create(true).append(true).open(path)
    }

    pub fn log(&self, level: Level, target: &str, msg: &str) {
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Send to channel; if thread panicked, we just drop the log
        let _ = self.sender.send(LogMessage {
            level,
            target: target.to_string(),
            msg: msg.to_string(),
            timestamp: ts,
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global logger singleton
// ─────────────────────────────────────────────────────────────────────────────

static LOGGER: OnceLock<NexusLogger> = OnceLock::new();

/// Initialize the global logger. Call once at startup.
/// - `verbose`: show Debug on console too
/// - `show_thinking`: show <think> blocks in console output
pub fn init(verbose: bool, show_thinking: bool) {
    LOGGER.get_or_init(|| NexusLogger::new(verbose, show_thinking));
}

/// Check if thinking should be shown on console.
pub fn show_thinking() -> bool {
    LOGGER.get().map(|l| l.show_thinking).unwrap_or(false)
}

/// Check if verbose mode is active.
pub fn is_verbose() -> bool {
    LOGGER.get().map(|l| l.verbose).unwrap_or(false)
}

/// Log a message at the given level. Use the macros instead of calling directly.
pub fn log(level: Level, target: &str, msg: &str) {
    if let Some(logger) = LOGGER.get() {
        logger.log(level, target, msg);
    }
}

/// Flush the log file. Call on clean shutdown.
pub fn flush() {
    // The background thread flushes when the channel drops (at process exit).
    // If we need an explicit flush, we would need a sync boundary.
    // For extreme performance, we rely on OS buffer flushing at exit.
}

// ─────────────────────────────────────────────────────────────────────────────
// Macros
// ─────────────────────────────────────────────────────────────────────────────

/// Log at TRACE level (file only).
#[macro_export]
macro_rules! nexus_trace {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::Level::Trace, module_path!(), &format!($($arg)*));
    };
}

/// Log at DEBUG level (file always; console in verbose mode).
#[macro_export]
macro_rules! nexus_debug {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::Level::Debug, module_path!(), &format!($($arg)*));
    };
}

/// Log at INFO level to file; also print to console with clean formatting.
/// Use this for the primary user-visible status messages.
#[macro_export]
macro_rules! nexus_info {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        $crate::logging::log($crate::logging::Level::Info, module_path!(), &msg);
    }};
}

/// Log at WARN level (file + console).
#[macro_export]
macro_rules! nexus_warn {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::Level::Warn, module_path!(), &format!($($arg)*));
    };
}

/// Log at ERROR level (file + console).
#[macro_export]
macro_rules! nexus_error {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::Level::Error, module_path!(), &format!($($arg)*));
    };
}
