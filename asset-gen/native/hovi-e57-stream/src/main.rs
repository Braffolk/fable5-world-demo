#![forbid(unsafe_code)]

mod args;
mod error;
mod input;
mod inspect;
mod json;
mod materialize;
mod probe;
mod provenance;
mod schema;

use crate::args::Command;
use crate::error::{AppError, AppResult};
use crate::json::{error_json, push_string};
use crate::provenance::{
    ERROR_SCHEMA, UPSTREAM_E57_CRATE_SHA256, UPSTREAM_E57_VERSION, VENDOR_PATCH_SET_SHA256,
};
use std::io::{self, Write};

fn main() {
    if let Err(error) = run() {
        let _ = io::stderr().write_all(error_json(&error).as_bytes());
        std::process::exit(error.code);
    }
}

fn run() -> AppResult<()> {
    match args::parse()? {
        Command::Version => write_stdout(&version_json()),
        Command::DryRun(arguments) => write_stdout(&inspect::dry_run(&arguments)?),
        Command::Probe(arguments) => write_stdout(&probe::probe(&arguments)?),
        Command::Materialize(arguments) => write_stdout(&materialize::materialize(&arguments)?),
        Command::Extract => Err(AppError::unavailable(
            "generic extraction is unavailable; use the authority-fixed materialize command",
        )),
    }
}

fn version_json() -> String {
    let mut output = String::from(
        "{\"schemaVersion\":\"laas-hovi-e57-native-version/1.0.0\",\"adapterVersion\":",
    );
    push_string(&mut output, env!("CARGO_PKG_VERSION"));
    output.push_str(",\"upstreamE57Version\":");
    push_string(&mut output, UPSTREAM_E57_VERSION);
    output.push_str(",\"upstreamCrateSha256\":");
    push_string(&mut output, UPSTREAM_E57_CRATE_SHA256);
    output.push_str(",\"vendorPatchSetSha256\":");
    push_string(&mut output, VENDOR_PATCH_SET_SHA256);
    output.push_str(",\"errorSchema\":");
    push_string(&mut output, ERROR_SCHEMA);
    output.push_str(",\"boundedProbeImplemented\":true,\"spatialMaterializationImplemented\":true,\"fullExtractionImplemented\":false}\n");
    output
}

fn write_stdout(payload: &str) -> AppResult<()> {
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(payload.as_bytes())
        .and_then(|_| stdout.flush())
        .map_err(|error| AppError::output(format!("failed to write stdout: {error}")))
}
