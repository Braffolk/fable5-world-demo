use crate::args::{InputSpec, MaterializeArgs};
use crate::error::{AppError, AppResult};
use crate::input::open_and_preflight;
use crate::json::{ensure_limit, push_string};
use crate::probe::{EXPECTED_SCANS, validate_authorized_source, validate_scan_set};
use crate::provenance::{UPSTREAM_E57_CRATE_SHA256, UPSTREAM_E57_VERSION, VENDOR_PATCH_SET_SHA256};
use e57::{E57Reader, PointCloud, RawValues, RecordName, RecordValue, Transform};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const REPORT_SCHEMA: &str = "laas-hovi-e57-spatial-materialization/1.0.0";
const RECORD_BYTES: u64 = 28;
const TICKS_PER_METRE: f64 = 40.0;
const SHARD_TICKS: i64 = 160;
const AOI_X_MIN: i64 = -271;
const AOI_X_MAX_EXCLUSIVE: i64 = 1491;
const AOI_Y_MIN: i64 = -244;
const AOI_Y_MAX_EXCLUSIVE: i64 = 1351;
const PROCESS_RESERVED_FILES: usize = 4;
const SHARD_BUFFER_BYTES: usize = 1 << 20;

const EXPECTED_POSE_BITS: [[u64; 7]; 16] = [
    [
        0x3fefffdb1494d92a,
        0,
        0,
        0xbf784df35c98ab0c,
        0,
        0,
        0x3ff8a3d70a3d70a6,
    ],
    [
        0xbfdc09d7dccbe104,
        0,
        0,
        0x3fecc4002b2d0900,
        0x4023af2f1143d586,
        0xbfe7a0cd88faadae,
        0x4000f615d188ee8a,
    ],
    [
        0x3febcf05e9fbc59f,
        0,
        0,
        0x3fdfaa62da57df67,
        0x4033fce7b70e0592,
        0xc0034e360f4ca42e,
        0x3ffb1f5a5faa6edb,
    ],
    [
        0x3fe5ca54ae52d59f,
        0,
        0,
        0xbfe76f41e69dbb56,
        0x403e240c16a6a9da,
        0xbfedee2d93bf39c5,
        0x3fe9ef2a1c76f573,
    ],
    [
        0x3fa71ebab44b8b1f,
        0,
        0,
        0x3feff7a4c3bc1f71,
        0x403e8b9508ba5f4e,
        0x40207c6e16dadbf9,
        0x3fe96b6da732e6aa,
    ],
    [
        0x3fe4be042e3a8a25,
        0,
        0,
        0xbfe85dfee756c5f5,
        0x4033bc2eef11e3ed,
        0x402194a6f5d47156,
        0x3ffce6acc6833f13,
    ],
    [
        0x3fe400627fed8d49,
        0,
        0,
        0x3fe8fa91e335bdd1,
        0x40234b1b959ef225,
        0x402386a5ae577dda,
        0x400365d5c51cdc46,
    ],
    [
        0x3fee867bc36da6cb,
        0,
        0,
        0xbfd33462c5811f9a,
        0x3fe41606a542de1c,
        0x4022c0ccb7228e0b,
        0x3ffe10a8eee5f1db,
    ],
    [
        0x3fe15f0d275ba171,
        0,
        0,
        0xbfeadfde7967d7c6,
        0x3ff8e226726952a3,
        0x403501adbc8c9e97,
        0x3ffc2c893df909b8,
    ],
    [
        0x3fecf8a70cfcdaf4,
        0,
        0,
        0x3fdb2d9f75c3f95d,
        0x402697b372c952fa,
        0x4033605b791c926d,
        0x40014b7e5fbbdad6,
    ],
    [
        0xbfccc43ab2a3bd46,
        0,
        0,
        0x3fef2e7104a63a0e,
        0x4034ea12abc88994,
        0x4031ee27e516bfec,
        0x3ff8ba3bbf1f03e9,
    ],
    [
        0x3fef584b3869feac,
        0,
        0,
        0x3fc9c4735ae10130,
        0x403ef54dcaa97b48,
        0x4030d7db444a212e,
        0x3fec99ea5746e10b,
    ],
    [
        0x3fec3cf885fc1920,
        0,
        0,
        0x3fde1b5be003ddea,
        0x40403ad138d84d52,
        0x403c197a4b7cb843,
        0x3fe32dc50a4aee56,
    ],
    [
        0xbfd46626576242a0,
        0,
        0,
        0x3fee54bcdb90a712,
        0x4035c46891b454f2,
        0x403c9e85c02634e1,
        0x3ffe56c977517fc8,
    ],
    [
        0x3fe95f0d4389842d,
        0,
        0,
        0x3fe3804f88f6eec1,
        0x4027c8aee8cbee8b,
        0x403d4404685571fd,
        0x3ff3094701b9a4b8,
    ],
    [
        0x3fecb4b3ec87e731,
        0,
        0,
        0x3fdc4848e4232cf6,
        0x3ffa4e4760469079,
        0x403cfc24e3b0b710,
        0x3ff31ba7d7ee3170,
    ],
];

pub fn materialize(arguments: &MaterializeArgs) -> AppResult<String> {
    if !matches!(arguments.reader.input, InputSpec::FileDescriptor(_)) {
        return Err(AppError::usage(
            "the authority-fixed materializer accepts only --input-fd",
        ));
    }
    let input = open_and_preflight(
        &arguments.reader.input,
        arguments.reader.max_file_bytes,
        arguments.reader.max_xml_bytes,
    )?;
    let source_mode = input.mode;
    let source_bytes = input.bytes;
    let mut reader = E57Reader::new(BufReader::with_capacity(1 << 20, input.file))
        .map_err(|error| AppError::validation(format!("E57 metadata parse failed: {error}")))?;
    let scans = reader.pointclouds();
    validate_scan_set(&scans, &arguments.reader, source_bytes)?;
    validate_authorized_source(&reader, &scans)?;
    validate_all_pose_bits(&scans)?;
    prepare_output(&arguments.output_dir, scans.len())?;

    let output_file_budget = arguments
        .max_open_files
        .checked_sub(PROCESS_RESERVED_FILES)
        .ok_or_else(|| {
            AppError::usage("open-file budget cannot reserve standard streams and input")
        })?;
    let mut sinks = SinkManager::new(
        arguments.output_dir.clone(),
        output_file_budget,
        arguments.max_output_bytes,
    );
    let mut scan_reports = Vec::with_capacity(scans.len());

    for (scan_ordinal, scan) in scans.iter().enumerate() {
        let indices = PrototypeIndices::from_scan(scan)?;
        let mut raw_reader = reader.pointcloud_raw(scan).map_err(|error| {
            AppError::validation(format!(
                "failed to open scan {scan_ordinal} raw iterator: {error}"
            ))
        })?;
        let mut values = RawValues::with_capacity(scan.prototype.len());
        let mut observed = ScanObserved::default();
        for source_ordinal in 0..scan.records {
            match raw_reader.next_into(&mut values) {
                Some(Ok(())) => {}
                Some(Err(error)) => {
                    return Err(AppError::validation(format!(
                        "scan {scan_ordinal} decode failed at source ordinal {source_ordinal}: {error}"
                    )));
                }
                None => {
                    return Err(AppError::validation(format!(
                        "scan {scan_ordinal} reached EOF before declared record {}",
                        scan.records
                    )));
                }
            }
            if values.len() != scan.prototype.len() {
                return Err(AppError::validation(format!(
                    "scan {scan_ordinal} record {source_ordinal} has {} values for a {}-field prototype",
                    values.len(),
                    scan.prototype.len()
                )));
            }
            let decoded = decode_record(scan_ordinal, source_ordinal, &values, scan, &indices)?;
            observed.decoded += 1;
            if decoded.intensity_finite {
                observed.finite_intensity += 1;
            } else {
                observed.non_finite_intensity += 1;
            }
            match decoded.destination {
                Destination::Point { shard_x, shard_y } => {
                    observed.valid_in_aoi += 1;
                    sinks.write(
                        SinkKey::Point {
                            scan: scan_ordinal as u8,
                            shard_x,
                            shard_y,
                        },
                        &decoded.bytes,
                    )?;
                }
                Destination::OutsideAoi => observed.valid_outside_aoi += 1,
                Destination::Nonpoint { invalid } => {
                    if invalid == 1 {
                        observed.direction += 1;
                    } else {
                        observed.invalid += 1;
                    }
                    sinks.write(
                        SinkKey::Nonpoint {
                            scan: scan_ordinal as u8,
                        },
                        &decoded.bytes,
                    )?;
                }
            }
        }
        raw_reader.finish().map_err(|error| {
            AppError::validation(format!(
                "scan {scan_ordinal} failed exact count/section completion: {error}"
            ))
        })?;
        if observed.decoded != scan.records {
            return Err(AppError::validation(format!(
                "scan {scan_ordinal} decoded {} records instead of {}",
                observed.decoded, scan.records
            )));
        }
        scan_reports.push(observed);
    }

    let output = sinks.finish()?;
    sync_output_directories(&arguments.output_dir, scans.len())?;
    render_report(
        arguments,
        source_mode,
        source_bytes,
        reader.guid(),
        &scans,
        &scan_reports,
        &output,
    )
}

fn validate_all_pose_bits(scans: &[PointCloud]) -> AppResult<()> {
    for (ordinal, (scan, expected)) in scans.iter().zip(EXPECTED_POSE_BITS).enumerate() {
        let transform = scan.transform.as_ref().ok_or_else(|| {
            AppError::validation(format!("authorized scan {ordinal} pose is absent"))
        })?;
        let actual = [
            transform.rotation.w,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ];
        if actual.map(f64::to_bits) != expected {
            return Err(AppError::validation(format!(
                "scan {ordinal} pose differs from the frozen source inventory"
            )));
        }
    }
    Ok(())
}

fn prepare_output(root: &Path, scan_count: usize) -> AppResult<()> {
    fs::create_dir(root).map_err(|error| {
        AppError::output(format!(
            "materialization output must be a new directory {}: {error}",
            root.display()
        ))
    })?;
    fs::create_dir(root.join("shards"))
        .and_then(|_| fs::create_dir(root.join("nonpoints")))
        .map_err(|error| AppError::output(format!("failed to create output layout: {error}")))?;
    for scan in 0..scan_count {
        fs::create_dir(root.join("shards").join(format!("scan-{scan:02}"))).map_err(|error| {
            AppError::output(format!("failed to create scan shard directory: {error}"))
        })?;
    }
    Ok(())
}

fn sync_output_directories(root: &Path, scan_count: usize) -> AppResult<()> {
    for scan in 0..scan_count {
        sync_directory(&root.join("shards").join(format!("scan-{scan:02}")))?;
    }
    sync_directory(&root.join("shards"))?;
    sync_directory(&root.join("nonpoints"))?;
    sync_directory(root)
}

fn sync_directory(path: &Path) -> AppResult<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            AppError::output(format!(
                "failed to sync output directory {}: {error}",
                path.display()
            ))
        })
}

struct PrototypeIndices {
    x: usize,
    y: usize,
    z: usize,
    intensity: usize,
    row: usize,
    column: usize,
    invalid: usize,
}

impl PrototypeIndices {
    fn from_scan(scan: &PointCloud) -> AppResult<Self> {
        let find = |name: RecordName| {
            scan.prototype
                .iter()
                .position(|record| record.name == name)
                .ok_or_else(|| {
                    AppError::validation(format!(
                        "validated scan is missing prototype field {name:?}"
                    ))
                })
        };
        Ok(Self {
            x: find(RecordName::CartesianX)?,
            y: find(RecordName::CartesianY)?,
            z: find(RecordName::CartesianZ)?,
            intensity: find(RecordName::Intensity)?,
            row: find(RecordName::RowIndex)?,
            column: find(RecordName::ColumnIndex)?,
            invalid: find(RecordName::CartesianInvalidState)?,
        })
    }
}

struct DecodedRecord {
    bytes: [u8; RECORD_BYTES as usize],
    destination: Destination,
    intensity_finite: bool,
}

enum Destination {
    Point { shard_x: i32, shard_y: i32 },
    OutsideAoi,
    Nonpoint { invalid: u8 },
}

fn decode_record(
    scan_ordinal: usize,
    source_ordinal: u64,
    values: &RawValues,
    scan: &PointCloud,
    indices: &PrototypeIndices,
) -> AppResult<DecodedRecord> {
    let raw_x = narrow_i32(
        scaled_integer(values, indices.x, "cartesianX")?,
        "cartesianX",
    )?;
    let raw_y = narrow_i32(
        scaled_integer(values, indices.y, "cartesianY")?,
        "cartesianY",
    )?;
    let raw_z = narrow_i32(
        scaled_integer(values, indices.z, "cartesianZ")?,
        "cartesianZ",
    )?;
    let source_ordinal = u32::try_from(source_ordinal)
        .map_err(|_| AppError::validation("source ordinal exceeds the 28-byte ABI"))?;
    let row = u16::try_from(integer(values, indices.row, "rowIndex")?)
        .map_err(|_| AppError::validation("rowIndex exceeds the 28-byte ABI"))?;
    let column = u16::try_from(integer(values, indices.column, "columnIndex")?)
        .map_err(|_| AppError::validation("columnIndex exceeds the 28-byte ABI"))?;
    let scan_ordinal = u8::try_from(scan_ordinal)
        .map_err(|_| AppError::validation("scan ordinal exceeds the 28-byte ABI"))?;
    let invalid = u8::try_from(integer(values, indices.invalid, "cartesianInvalidState")?)
        .map_err(|_| AppError::validation("cartesianInvalidState exceeds the 28-byte ABI"))?;
    if invalid > 2 {
        return Err(AppError::validation(format!(
            "cartesianInvalidState contains out-of-domain value {invalid}"
        )));
    }
    let intensity = match values[indices.intensity] {
        RecordValue::Single(value) => value,
        _ => {
            return Err(AppError::validation(
                "intensity raw value does not match the Single prototype",
            ));
        }
    };
    let intensity_finite = intensity.is_finite();
    let flags = u16::from(intensity_finite);
    let destination = if invalid == 0 {
        let local = [
            coordinate(values, scan, indices.x, "cartesianX")?,
            coordinate(values, scan, indices.y, "cartesianY")?,
            coordinate(values, scan, indices.z, "cartesianZ")?,
        ];
        let file = apply_pose(
            scan.transform
                .as_ref()
                .ok_or_else(|| AppError::validation("validated scan lost its pose"))?,
            local,
        )?;
        match (metre_tick(file[0])?, metre_tick(file[1])?) {
            (x, y)
                if (AOI_X_MIN..AOI_X_MAX_EXCLUSIVE).contains(&x)
                    && (AOI_Y_MIN..AOI_Y_MAX_EXCLUSIVE).contains(&y) =>
            {
                Destination::Point {
                    shard_x: i32::try_from((x - AOI_X_MIN).div_euclid(SHARD_TICKS))
                        .map_err(|_| AppError::validation("X shard index exceeds i32"))?,
                    shard_y: i32::try_from((y - AOI_Y_MIN).div_euclid(SHARD_TICKS))
                        .map_err(|_| AppError::validation("Y shard index exceeds i32"))?,
                }
            }
            _ => Destination::OutsideAoi,
        }
    } else {
        Destination::Nonpoint { invalid }
    };

    let mut bytes = [0_u8; RECORD_BYTES as usize];
    bytes[0..4].copy_from_slice(&source_ordinal.to_le_bytes());
    bytes[4..8].copy_from_slice(&raw_x.to_le_bytes());
    bytes[8..12].copy_from_slice(&raw_y.to_le_bytes());
    bytes[12..16].copy_from_slice(&raw_z.to_le_bytes());
    bytes[16..20].copy_from_slice(&intensity.to_bits().to_le_bytes());
    bytes[20..22].copy_from_slice(&row.to_le_bytes());
    bytes[22..24].copy_from_slice(&column.to_le_bytes());
    bytes[24] = scan_ordinal;
    bytes[25] = invalid;
    bytes[26..28].copy_from_slice(&flags.to_le_bytes());
    Ok(DecodedRecord {
        bytes,
        destination,
        intensity_finite,
    })
}

fn coordinate(values: &RawValues, scan: &PointCloud, index: usize, label: &str) -> AppResult<f64> {
    let value = values[index]
        .to_f64(&scan.prototype[index].data_type)
        .map_err(|error| AppError::validation(format!("{label} conversion failed: {error}")))?;
    if !value.is_finite() {
        return Err(AppError::validation(format!("{label} is non-finite")));
    }
    Ok(value)
}

fn apply_pose(transform: &Transform, point: [f64; 3]) -> AppResult<[f64; 3]> {
    let q = &transform.rotation;
    let t = &transform.translation;
    let qv = [q.x, q.y, q.z];
    let uv = cross(qv, point);
    let uuv = cross(qv, uv);
    let result = [
        point[0] + 2.0 * (q.w * uv[0] + uuv[0]) + t.x,
        point[1] + 2.0 * (q.w * uv[1] + uuv[1]) + t.y,
        point[2] + 2.0 * (q.w * uv[2] + uuv[2]) + t.z,
    ];
    if result.iter().any(|value| !value.is_finite()) {
        return Err(AppError::validation(
            "publisher pose produced a non-finite file-frame coordinate",
        ));
    }
    Ok(result)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn metre_tick(value: f64) -> AppResult<i64> {
    let tick = (value * TICKS_PER_METRE).floor();
    if !tick.is_finite() || tick < i64::MIN as f64 || tick > i64::MAX as f64 {
        return Err(AppError::validation("file-frame AOI tick exceeds i64"));
    }
    Ok(tick as i64)
}

fn scaled_integer(values: &RawValues, index: usize, label: &str) -> AppResult<i64> {
    match values[index] {
        RecordValue::ScaledInteger(value) => Ok(value),
        _ => Err(AppError::validation(format!(
            "{label} raw value does not match its ScaledInteger prototype"
        ))),
    }
}

fn integer(values: &RawValues, index: usize, label: &str) -> AppResult<i64> {
    match values[index] {
        RecordValue::Integer(value) => Ok(value),
        _ => Err(AppError::validation(format!(
            "{label} raw value does not match its Integer prototype"
        ))),
    }
}

fn narrow_i32(value: i64, label: &str) -> AppResult<i32> {
    i32::try_from(value)
        .map_err(|_| AppError::validation(format!("{label} exceeds the 28-byte ABI")))
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum SinkKey {
    Point {
        scan: u8,
        shard_x: i32,
        shard_y: i32,
    },
    Nonpoint {
        scan: u8,
    },
}

struct SinkState {
    relative_path: String,
    records: u64,
    bytes: u64,
    hasher: Sha256,
    last_use: u64,
}

struct SinkManager {
    root: PathBuf,
    open_limit: usize,
    max_bytes: u64,
    total_bytes: u64,
    clock: u64,
    states: HashMap<SinkKey, SinkState>,
    open: HashMap<SinkKey, BufWriter<File>>,
}

impl SinkManager {
    fn new(root: PathBuf, open_limit: usize, max_bytes: u64) -> Self {
        Self {
            root,
            open_limit,
            max_bytes,
            total_bytes: 0,
            clock: 0,
            states: HashMap::new(),
            open: HashMap::new(),
        }
    }

    fn write(&mut self, key: SinkKey, record: &[u8; RECORD_BYTES as usize]) -> AppResult<()> {
        let next_total = self
            .total_bytes
            .checked_add(RECORD_BYTES)
            .ok_or_else(|| AppError::output("materialized output byte count overflow"))?;
        if next_total > self.max_bytes {
            return Err(AppError::output(format!(
                "materialized output exceeds the {}-byte hard command limit",
                self.max_bytes
            )));
        }
        if !self.states.contains_key(&key) {
            self.states.insert(
                key.clone(),
                SinkState {
                    relative_path: relative_path(&key),
                    records: 0,
                    bytes: 0,
                    hasher: Sha256::new(),
                    last_use: 0,
                },
            );
        }
        self.ensure_open(&key)?;
        self.open
            .get_mut(&key)
            .ok_or_else(|| AppError::output("output LRU lost an open shard"))?
            .write_all(record)
            .map_err(|error| AppError::output(format!("failed writing output record: {error}")))?;
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| AppError::output("output LRU clock overflow"))?;
        let state = self
            .states
            .get_mut(&key)
            .ok_or_else(|| AppError::output("output LRU lost shard state"))?;
        state.records += 1;
        state.bytes += RECORD_BYTES;
        state.hasher.update(record);
        state.last_use = self.clock;
        self.total_bytes = next_total;
        Ok(())
    }

    fn ensure_open(&mut self, key: &SinkKey) -> AppResult<()> {
        if self.open.contains_key(key) {
            return Ok(());
        }
        if self.open.len() >= self.open_limit {
            let evict = self
                .open
                .keys()
                .min_by_key(|candidate| {
                    self.states.get(*candidate).map_or(u64::MAX, |s| s.last_use)
                })
                .cloned()
                .ok_or_else(|| AppError::output("output LRU could not select an eviction"))?;
            let mut file = self
                .open
                .remove(&evict)
                .ok_or_else(|| AppError::output("output LRU eviction lost its file"))?;
            file.flush().map_err(|error| {
                AppError::output(format!("failed flushing evicted shard: {error}"))
            })?;
        }
        let state = self
            .states
            .get(key)
            .ok_or_else(|| AppError::output("output LRU lost shard state before open"))?;
        let path = self.root.join(&state.relative_path);
        let file = if state.records == 0 {
            OpenOptions::new().write(true).create_new(true).open(&path)
        } else {
            OpenOptions::new().append(true).open(&path)
        }
        .map_err(|error| {
            AppError::output(format!(
                "failed opening output shard {}: {error}",
                path.display()
            ))
        })?;
        self.open.insert(
            key.clone(),
            BufWriter::with_capacity(SHARD_BUFFER_BYTES, file),
        );
        Ok(())
    }

    fn finish(mut self) -> AppResult<MaterializedOutput> {
        for (_, mut file) in self.open.drain() {
            file.flush().map_err(|error| {
                AppError::output(format!("failed flushing output shard: {error}"))
            })?;
        }
        let mut entries: Vec<_> = self.states.into_iter().collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let mut artifacts = Vec::with_capacity(entries.len());
        for (key, state) in entries {
            let path = self.root.join(&state.relative_path);
            let file = File::open(&path).map_err(|error| {
                AppError::output(format!(
                    "failed reopening output shard {}: {error}",
                    path.display()
                ))
            })?;
            let bytes = file
                .metadata()
                .and_then(|metadata| file.sync_all().map(|_| metadata.len()))
                .map_err(|error| {
                    AppError::output(format!(
                        "failed syncing output shard {}: {error}",
                        path.display()
                    ))
                })?;
            if bytes != state.bytes || bytes != state.records * RECORD_BYTES {
                return Err(AppError::output(format!(
                    "output shard {} has {bytes} bytes instead of {}",
                    path.display(),
                    state.bytes
                )));
            }
            artifacts.push(Artifact {
                key,
                relative_path: state.relative_path,
                records: state.records,
                bytes,
                sha256: hex(&state.hasher.finalize()),
            });
        }
        Ok(MaterializedOutput {
            bytes: self.total_bytes,
            artifacts,
        })
    }
}

fn relative_path(key: &SinkKey) -> String {
    match key {
        SinkKey::Point {
            scan,
            shard_x,
            shard_y,
        } => format!("shards/scan-{scan:02}/x{shard_x:+04}_y{shard_y:+04}.bin"),
        SinkKey::Nonpoint { scan } => format!("nonpoints/scan-{scan:02}.bin"),
    }
}

struct Artifact {
    key: SinkKey,
    relative_path: String,
    records: u64,
    bytes: u64,
    sha256: String,
}

struct MaterializedOutput {
    bytes: u64,
    artifacts: Vec<Artifact>,
}

#[derive(Default)]
struct ScanObserved {
    decoded: u64,
    valid_in_aoi: u64,
    valid_outside_aoi: u64,
    direction: u64,
    invalid: u64,
    finite_intensity: u64,
    non_finite_intensity: u64,
}

fn render_report(
    arguments: &MaterializeArgs,
    source_mode: &str,
    source_bytes: u64,
    file_guid: &str,
    scans: &[PointCloud],
    observed: &[ScanObserved],
    output: &MaterializedOutput,
) -> AppResult<String> {
    let mut json = String::with_capacity(512 * 1024);
    json.push_str("{\"schemaVersion\":");
    push_string(&mut json, REPORT_SCHEMA);
    json.push_str(",\"status\":\"spatial_materialization_complete\",\"profile\":\"hy-spruce4\",\"reader\":{\"adapterVersion\":");
    push_string(&mut json, env!("CARGO_PKG_VERSION"));
    json.push_str(",\"upstreamE57Version\":");
    push_string(&mut json, UPSTREAM_E57_VERSION);
    json.push_str(",\"upstreamCrateSha256\":");
    push_string(&mut json, UPSTREAM_E57_CRATE_SHA256);
    json.push_str(",\"vendorPatchSetSha256\":");
    push_string(&mut json, VENDOR_PATCH_SET_SHA256);
    write!(
        json,
        ",\"exactCountAndSectionEofValidated\":true}},\"source\":{{\"inputMode\":\"{source_mode}\",\"bytes\":\"{source_bytes}\",\"digestComputed\":false,\"fileGuid\":"
    )
    .map_err(|_| AppError::output("failed to format materialization source"))?;
    push_string(&mut json, file_guid);
    json.push_str("},\"spatialContract\":{\"coordinateFrame\":\"publisher_file_frame_for_assignment_only\",\"storedCoordinates\":\"raw_scan_local_scaled_integers\",\"tickMetres\":0.025,\"aoiTicks\":{\"xMinInclusive\":-271,\"xMaxExclusive\":1491,\"yMinInclusive\":-244,\"yMaxExclusive\":1351},\"shardTicks\":160,\"shardMetres\":4.0,\"shardIndexing\":\"aoi_local_euclidean_floor\"},\"recordAbi\":{\"byteOrder\":\"little_endian\",\"bytesPerRecord\":28,\"fields\":[\"sourceOrdinal:u32@0\",\"rawX:i32@4\",\"rawY:i32@8\",\"rawZ:i32@12\",\"rawIntensitySingleBits:u32@16\",\"row:u16@20\",\"column:u16@22\",\"scanOrdinal:u8@24\",\"cartesianInvalidState:u8@25\",\"flags:u16@26\"],\"flags\":{\"bit0\":\"intensity_finite\",\"allOtherBits\":\"zero\"}},\"scans\":[");
    for (ordinal, (scan, counts)) in scans.iter().zip(observed).enumerate() {
        if ordinal > 0 {
            json.push(',');
        }
        write!(json, "{{\"ordinal\":{ordinal},\"guid\":")
            .map_err(|_| AppError::output("failed to format scan report"))?;
        push_string(&mut json, scan.guid.as_deref().unwrap_or_default());
        write!(
            json,
            ",\"publisherDeclaredRecords\":\"{}\",\"decodedRecords\":\"{}\",\"validInAoi\":\"{}\",\"validOutsideAoi\":\"{}\",\"directionRecords\":\"{}\",\"invalidRecords\":\"{}\",\"finiteIntensity\":\"{}\",\"nonFiniteIntensity\":\"{}\",\"poseAppliedToStoredCoordinates\":false,\"poseAppliedForAoiAssignment\":true}}",
            scan.records,
            counts.decoded,
            counts.valid_in_aoi,
            counts.valid_outside_aoi,
            counts.direction,
            counts.invalid,
            counts.finite_intensity,
            counts.non_finite_intensity,
        )
        .map_err(|_| AppError::output("failed to format scan counters"))?;
    }
    json.push_str("],\"output\":{\"layoutRoot\":\".\"");
    write!(
        json,
        ",\"bytes\":\"{}\",\"artifactCount\":{},\"atomicPublication\":false,\"stagingOnly\":true,\"artifacts\":[",
        output.bytes,
        output.artifacts.len()
    )
    .map_err(|_| AppError::output("failed to format output totals"))?;
    for (index, artifact) in output.artifacts.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str("{\"kind\":");
        push_string(
            &mut json,
            match artifact.key {
                SinkKey::Point { .. } => "point_shard",
                SinkKey::Nonpoint { .. } => "nonpoint_stream",
            },
        );
        match artifact.key {
            SinkKey::Point {
                scan,
                shard_x,
                shard_y,
            } => write!(
                json,
                ",\"scan\":{scan},\"shardX\":{shard_x},\"shardY\":{shard_y},\"path\":"
            ),
            SinkKey::Nonpoint { scan } => {
                write!(json, ",\"scan\":{scan},\"path\":")
            }
        }
        .map_err(|_| AppError::output("failed to format artifact key"))?;
        push_string(&mut json, &artifact.relative_path);
        write!(
            json,
            ",\"records\":\"{}\",\"bytes\":\"{}\",\"sha256\":\"{}\"}}",
            artifact.records, artifact.bytes, artifact.sha256
        )
        .map_err(|_| AppError::output("failed to format artifact identity"))?;
    }
    writeln!(
        json,
        "]}},\"limits\":{{\"maxFileBytes\":\"{}\",\"maxXmlBytes\":\"{}\",\"maxScans\":{},\"maxPrototypeFields\":{},\"maxDeclaredRecords\":\"{}\",\"maxJsonBytes\":{},\"maxOutputBytes\":\"{}\",\"maxOpenFiles\":{},\"maxOpenOutputFiles\":{}}},\"result\":{{\"allScansDecodedOnce\":true,\"scanCount\":{},\"publisherRecordCount\":\"{}\",\"recordsOutsideAoiDropped\":true,\"invalidAndDirectionRetained\":true,\"groundFiltered\":false,\"surfaceClaim\":false,\"targetTruth\":false,\"synthesisAuthorized\":false}}}}",
        arguments.reader.max_file_bytes,
        arguments.reader.max_xml_bytes,
        arguments.reader.max_scans,
        arguments.reader.max_prototype_fields,
        arguments.reader.max_declared_records,
        arguments.reader.max_json_bytes,
        arguments.max_output_bytes,
        arguments.max_open_files,
        arguments.max_open_files - PROCESS_RESERVED_FILES,
        scans.len(),
        EXPECTED_SCANS.iter().map(|scan| scan.1).sum::<u64>(),
    )
    .map_err(|_| AppError::output("failed to format materialization result"))?;
    ensure_limit(&json, arguments.reader.max_json_bytes)?;
    Ok(json)
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use e57::{Quaternion, Translation};

    #[test]
    fn aoi_local_shards_cover_the_frozen_half_open_domain() {
        assert_eq!((-271_i64 - AOI_X_MIN).div_euclid(SHARD_TICKS), 0);
        assert_eq!((-111_i64 - AOI_X_MIN).div_euclid(SHARD_TICKS), 1);
        assert_eq!((1490_i64 - AOI_X_MIN).div_euclid(SHARD_TICKS), 11);
        assert_eq!((-244_i64 - AOI_Y_MIN).div_euclid(SHARD_TICKS), 0);
        assert_eq!((1350_i64 - AOI_Y_MIN).div_euclid(SHARD_TICKS), 9);
    }

    #[test]
    fn publisher_pose_rotation_and_translation_match_e57_formula() {
        let transform = Transform {
            rotation: Quaternion {
                w: 0.0,
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            translation: Translation {
                x: 10.0,
                y: 20.0,
                z: 30.0,
            },
        };
        assert_eq!(
            apply_pose(&transform, [2.0, 3.0, 4.0]).unwrap(),
            [8.0, 17.0, 34.0]
        );
    }
}
