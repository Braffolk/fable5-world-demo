use crate::error::{AppError, AppResult};
use std::ffi::OsString;
use std::path::PathBuf;

const HARD_MAX_FILE_BYTES: u64 = 1_u64 << 40;
const HARD_MAX_XML_BYTES: u64 = 8 * 1024 * 1024;
const HARD_MAX_SCANS: usize = 4096;
const HARD_MAX_PROTOTYPE_FIELDS: usize = 1024;
const HARD_MAX_JSON_BYTES: usize = 32 * 1024 * 1024;
const HARD_MAX_MATERIALIZED_BYTES: u64 = 120 * 1024 * 1024 * 1024;
const HARD_MAX_OPEN_FILES: usize = 24;

#[derive(Debug)]
pub enum Command {
    Version,
    DryRun(DryRunArgs),
    Probe(ProbeArgs),
    Materialize(MaterializeArgs),
    Extract,
}

#[derive(Debug)]
pub enum InputSpec {
    Path(PathBuf),
    FileDescriptor(u32),
}

#[derive(Debug)]
pub struct DryRunArgs {
    pub input: InputSpec,
    pub max_file_bytes: u64,
    pub max_xml_bytes: u64,
    pub max_scans: usize,
    pub max_prototype_fields: usize,
    pub max_declared_records: u64,
    pub max_json_bytes: usize,
}

#[derive(Debug)]
pub struct ProbeArgs {
    pub reader: DryRunArgs,
    pub output_fd: u32,
}

#[derive(Debug)]
pub struct MaterializeArgs {
    pub reader: DryRunArgs,
    pub output_dir: PathBuf,
    pub max_output_bytes: u64,
    pub max_open_files: usize,
}

pub fn parse() -> AppResult<Command> {
    let mut arguments = std::env::args_os();
    let _program = arguments.next();
    let command = arguments.next().ok_or_else(|| {
        AppError::usage("expected one of: version, dry-run, probe, materialize, extract")
    })?;
    let command = command
        .to_str()
        .ok_or_else(|| AppError::usage("command must be valid UTF-8"))?;
    let remaining: Vec<OsString> = arguments.collect();
    match command {
        "version" => {
            require_empty(&remaining, "version")?;
            Ok(Command::Version)
        }
        "probe" => Ok(Command::Probe(parse_probe(remaining)?)),
        "materialize" => Ok(Command::Materialize(parse_materialize(remaining)?)),
        "extract" => {
            require_empty(&remaining, "extract")?;
            Ok(Command::Extract)
        }
        "dry-run" => Ok(Command::DryRun(parse_dry_run(remaining)?)),
        _ => Err(AppError::usage(format!("unknown command: {command}"))),
    }
}

fn parse_materialize(arguments: Vec<OsString>) -> AppResult<MaterializeArgs> {
    let mut output_dir = None;
    let mut max_output_bytes = None;
    let mut max_open_files = None;
    let mut reader_arguments = Vec::with_capacity(arguments.len());
    let mut index = 0;
    while index < arguments.len() {
        let option = utf8(&arguments[index], "option")?;
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| AppError::usage(format!("missing value for {option}")))?;
        match option {
            "--output-dir" => set_once(&mut output_dir, PathBuf::from(value), "output-dir")?,
            "--max-output-bytes" => set_once(
                &mut max_output_bytes,
                parse_u64(value, option)?,
                "max-output-bytes",
            )?,
            "--max-open-files" => set_once(
                &mut max_open_files,
                parse_usize(value, option)?,
                "max-open-files",
            )?,
            _ => {
                reader_arguments.push(arguments[index].clone());
                reader_arguments.push(value.clone());
            }
        }
        index += 2;
    }
    let parsed = MaterializeArgs {
        reader: parse_dry_run(reader_arguments)?,
        output_dir: required(output_dir, "--output-dir")?,
        max_output_bytes: required(max_output_bytes, "--max-output-bytes")?,
        max_open_files: required(max_open_files, "--max-open-files")?,
    };
    if parsed.max_output_bytes == 0 || parsed.max_output_bytes > HARD_MAX_MATERIALIZED_BYTES {
        return Err(AppError::usage(format!(
            "--max-output-bytes must be within 1..={HARD_MAX_MATERIALIZED_BYTES}"
        )));
    }
    if !(5..=HARD_MAX_OPEN_FILES).contains(&parsed.max_open_files) {
        return Err(AppError::usage(format!(
            "--max-open-files must be within 5..={HARD_MAX_OPEN_FILES}"
        )));
    }
    Ok(parsed)
}

fn parse_probe(arguments: Vec<OsString>) -> AppResult<ProbeArgs> {
    let mut output_fd = None;
    let mut reader_arguments = Vec::with_capacity(arguments.len());
    let mut index = 0;
    while index < arguments.len() {
        let option = utf8(&arguments[index], "option")?;
        if option == "--output-fd" {
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| AppError::usage("missing value for --output-fd"))?;
            set_once(
                &mut output_fd,
                parse_u32(value, "--output-fd")?,
                "output-fd",
            )?;
            index += 2;
        } else {
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| AppError::usage(format!("missing value for {option}")))?;
            reader_arguments.push(arguments[index].clone());
            reader_arguments.push(value.clone());
            index += 2;
        }
    }
    let reader = parse_dry_run(reader_arguments)?;
    let output_fd = required(output_fd, "--output-fd")?;
    if !(3..=1024).contains(&output_fd) {
        return Err(AppError::usage("--output-fd must be within 3..=1024"));
    }
    if matches!(&reader.input, InputSpec::FileDescriptor(input_fd) if *input_fd == output_fd) {
        return Err(AppError::usage(
            "--input-fd and --output-fd must be different descriptors",
        ));
    }
    Ok(ProbeArgs { reader, output_fd })
}

fn require_empty(arguments: &[OsString], command: &str) -> AppResult<()> {
    if arguments.is_empty() {
        Ok(())
    } else {
        Err(AppError::usage(format!(
            "{command} does not accept arguments"
        )))
    }
}

fn parse_dry_run(arguments: Vec<OsString>) -> AppResult<DryRunArgs> {
    let mut input: Option<InputSpec> = None;
    let mut max_file_bytes = None;
    let mut max_xml_bytes = None;
    let mut max_scans = None;
    let mut max_prototype_fields = None;
    let mut max_declared_records = None;
    let mut max_json_bytes = None;
    let mut profile_seen = false;

    let mut index = 0;
    while index < arguments.len() {
        let option = utf8(&arguments[index], "option")?;
        index += 1;
        let value = arguments
            .get(index)
            .ok_or_else(|| AppError::usage(format!("missing value for {option}")))?;
        index += 1;
        match option {
            "--input" => set_once(&mut input, InputSpec::Path(PathBuf::from(value)), "input")?,
            "--input-fd" => set_once(
                &mut input,
                InputSpec::FileDescriptor(parse_u32(value, option)?),
                "input",
            )?,
            "--profile" => {
                if profile_seen {
                    return Err(AppError::usage("duplicate --profile"));
                }
                profile_seen = true;
                if utf8(value, option)? != "hy-spruce4" {
                    return Err(AppError::usage("the only supported profile is hy-spruce4"));
                }
            }
            "--max-file-bytes" => set_once(
                &mut max_file_bytes,
                parse_u64(value, option)?,
                "max-file-bytes",
            )?,
            "--max-xml-bytes" => set_once(
                &mut max_xml_bytes,
                parse_u64(value, option)?,
                "max-xml-bytes",
            )?,
            "--max-scans" => set_once(&mut max_scans, parse_usize(value, option)?, "max-scans")?,
            "--max-prototype-fields" => set_once(
                &mut max_prototype_fields,
                parse_usize(value, option)?,
                "max-prototype-fields",
            )?,
            "--max-declared-records" => set_once(
                &mut max_declared_records,
                parse_u64(value, option)?,
                "max-declared-records",
            )?,
            "--max-json-bytes" => set_once(
                &mut max_json_bytes,
                parse_usize(value, option)?,
                "max-json-bytes",
            )?,
            _ => return Err(AppError::usage(format!("unknown option: {option}"))),
        }
    }

    if !profile_seen {
        return Err(AppError::usage("missing required --profile hy-spruce4"));
    }
    let parsed = DryRunArgs {
        input: input.ok_or_else(|| AppError::usage("missing --input or --input-fd"))?,
        max_file_bytes: required(max_file_bytes, "--max-file-bytes")?,
        max_xml_bytes: required(max_xml_bytes, "--max-xml-bytes")?,
        max_scans: required(max_scans, "--max-scans")?,
        max_prototype_fields: required(max_prototype_fields, "--max-prototype-fields")?,
        max_declared_records: required(max_declared_records, "--max-declared-records")?,
        max_json_bytes: required(max_json_bytes, "--max-json-bytes")?,
    };
    validate_hard_limits(&parsed)?;
    Ok(parsed)
}

fn validate_hard_limits(arguments: &DryRunArgs) -> AppResult<()> {
    if arguments.max_file_bytes == 0 || arguments.max_file_bytes > HARD_MAX_FILE_BYTES {
        return Err(AppError::usage(format!(
            "--max-file-bytes must be within 1..={HARD_MAX_FILE_BYTES}"
        )));
    }
    if arguments.max_xml_bytes == 0 || arguments.max_xml_bytes > HARD_MAX_XML_BYTES {
        return Err(AppError::usage(format!(
            "--max-xml-bytes must be within 1..={HARD_MAX_XML_BYTES}"
        )));
    }
    if arguments.max_scans == 0 || arguments.max_scans > HARD_MAX_SCANS {
        return Err(AppError::usage(format!(
            "--max-scans must be within 1..={HARD_MAX_SCANS}"
        )));
    }
    if arguments.max_prototype_fields == 0
        || arguments.max_prototype_fields > HARD_MAX_PROTOTYPE_FIELDS
    {
        return Err(AppError::usage(format!(
            "--max-prototype-fields must be within 1..={HARD_MAX_PROTOTYPE_FIELDS}"
        )));
    }
    if arguments.max_declared_records == 0 {
        return Err(AppError::usage("--max-declared-records must be positive"));
    }
    if arguments.max_json_bytes == 0 || arguments.max_json_bytes > HARD_MAX_JSON_BYTES {
        return Err(AppError::usage(format!(
            "--max-json-bytes must be within 1..={HARD_MAX_JSON_BYTES}"
        )));
    }
    if let InputSpec::FileDescriptor(descriptor) = &arguments.input
        && !(3..=1024).contains(descriptor)
    {
        return Err(AppError::usage("--input-fd must be within 3..=1024"));
    }
    Ok(())
}

fn set_once<T>(slot: &mut Option<T>, value: T, label: &str) -> AppResult<()> {
    if slot.replace(value).is_some() {
        Err(AppError::usage(format!("duplicate {label}")))
    } else {
        Ok(())
    }
}

fn required<T>(slot: Option<T>, option: &str) -> AppResult<T> {
    slot.ok_or_else(|| AppError::usage(format!("missing required {option}")))
}

fn utf8<'a>(value: &'a OsString, label: &str) -> AppResult<&'a str> {
    value
        .to_str()
        .ok_or_else(|| AppError::usage(format!("{label} must be valid UTF-8")))
}

fn parse_u64(value: &OsString, option: &str) -> AppResult<u64> {
    utf8(value, option)?
        .parse::<u64>()
        .map_err(|_| AppError::usage(format!("{option} must be an unsigned integer")))
}

fn parse_u32(value: &OsString, option: &str) -> AppResult<u32> {
    utf8(value, option)?
        .parse::<u32>()
        .map_err(|_| AppError::usage(format!("{option} must be an unsigned integer")))
}

fn parse_usize(value: &OsString, option: &str) -> AppResult<usize> {
    utf8(value, option)?
        .parse::<usize>()
        .map_err(|_| AppError::usage(format!("{option} must be an unsigned integer")))
}
