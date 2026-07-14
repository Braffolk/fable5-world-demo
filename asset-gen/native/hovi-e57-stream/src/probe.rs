use crate::args::{DryRunArgs, InputSpec, ProbeArgs};
use crate::error::{AppError, AppResult};
use crate::input::open_and_preflight;
use crate::json::{ensure_limit, push_string};
use crate::provenance::{UPSTREAM_E57_CRATE_SHA256, UPSTREAM_E57_VERSION, VENDOR_PATCH_SET_SHA256};
use crate::schema::{record_name, validate_scan};
use e57::{E57Reader, PointCloud, RawValues, RecordDataType, RecordName, RecordValue};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;

const PROBE_SCHEMA: &str = "laas-hovi-e57-native-probe/1.0.0";
const SCAN_ORDINAL: usize = 0;
const EXPECTED_GUID: &str = "0000000000000001";
const START_ORDINAL: u64 = 0;
const STOP_ORDINAL_EXCLUSIVE: u64 = 65_536;
const CANONICAL_RECORD_BYTES: usize = 96;
const EXPECTED_FILE_GUID: &str = "{C70845D7-08E0-4185-974C-EF84E9EDEA55}";
pub(crate) const EXPECTED_SCANS: [(&str, u64, u64); 16] = [
    ("0000000000000001", 245_788_993, 48),
    ("0000000000000002", 266_939_657, 4_906_634_044),
    ("0000000000000003", 265_134_729, 10_235_477_052),
    ("0000000000000004", 262_355_367, 15_528_290_144),
    ("0000000000000005", 271_090_672, 20_765_621_704),
    ("0000000000000006", 271_783_669, 26_177_327_196),
    ("0000000000000007", 269_210_517, 31_602_866_312),
    ("0000000000000008", 256_875_317, 36_977_040_196),
    ("0000000000000009", 258_486_083, 42_104_979_120),
    ("000000000000000a", 265_325_952, 47_265_072_096),
    ("000000000000000b", 271_817_243, 52_561_702_392),
    ("000000000000000c", 273_796_683, 57_987_911_724),
    ("000000000000000d", 269_993_554, 63_453_634_548),
    ("000000000000000e", 255_020_027, 68_843_439_396),
    ("000000000000000f", 273_897_266, 73_934_343_032),
    ("0000000000000010", 271_281_592, 79_402_073_724),
];

pub fn probe(arguments: &ProbeArgs) -> AppResult<String> {
    let reader_arguments = &arguments.reader;
    if !matches!(reader_arguments.input, InputSpec::FileDescriptor(_)) {
        return Err(AppError::usage(
            "the authority-fixed probe accepts only --input-fd",
        ));
    }
    let input = open_and_preflight(
        &reader_arguments.input,
        reader_arguments.max_file_bytes,
        reader_arguments.max_xml_bytes,
    )?;
    let mut output_file = open_output(arguments.output_fd, input.device, input.inode)?;
    let source_mode = input.mode;
    let source_bytes = input.bytes;
    let mut reader = E57Reader::new(BufReader::with_capacity(1 << 20, input.file))
        .map_err(|error| AppError::validation(format!("E57 metadata parse failed: {error}")))?;
    let scans = reader.pointclouds();
    validate_scan_set(&scans, reader_arguments, source_bytes)?;
    validate_authorized_source(&reader, &scans)?;
    let scan = scans
        .get(SCAN_ORDINAL)
        .ok_or_else(|| AppError::validation("authorized probe scan ordinal 0 is absent"))?
        .clone();
    let guid = scan.guid.as_deref().unwrap_or_default();
    if guid != EXPECTED_GUID {
        return Err(AppError::validation(format!(
            "scan 0 GUID {guid:?} differs from authorized GUID {EXPECTED_GUID}"
        )));
    }
    if scan.records < STOP_ORDINAL_EXCLUSIVE {
        return Err(AppError::validation(format!(
            "scan 0 declares only {} records; probe requires source ordinals 0 through {}",
            scan.records,
            STOP_ORDINAL_EXCLUSIVE - 1,
        )));
    }

    let indices = PrototypeIndices::from_scan(&scan)?;
    let mut raw_reader = reader.pointcloud_raw(&scan).map_err(|error| {
        AppError::validation(format!("failed to open scan 0 raw iterator: {error}"))
    })?;
    let mut values = RawValues::with_capacity(scan.prototype.len());
    let mut observed = Observed::default();
    let mut hasher = Sha256::new();

    for ordinal in START_ORDINAL..STOP_ORDINAL_EXCLUSIVE {
        match raw_reader.next_into(&mut values) {
            Some(Ok(())) => {}
            Some(Err(error)) => {
                return Err(AppError::validation(format!(
                    "scan 0 decode failed at source ordinal {ordinal}: {error}"
                )));
            }
            None => {
                return Err(AppError::validation(format!(
                    "scan 0 reached EOF before authorized stop ordinal {STOP_ORDINAL_EXCLUSIVE}"
                )));
            }
        }
        if values.len() != scan.prototype.len() {
            return Err(AppError::validation(format!(
                "scan 0 record {ordinal} has {} values for a {}-field prototype",
                values.len(),
                scan.prototype.len()
            )));
        }
        let record = observe_record(ordinal, &values, &scan, &indices, &mut observed)?;
        output_file.write_all(&record).map_err(|error| {
            AppError::output(format!(
                "failed writing canonical probe record {ordinal}: {error}"
            ))
        })?;
        hasher.update(record);
    }

    output_file
        .flush()
        .and_then(|_| output_file.sync_all())
        .map_err(|error| AppError::output(format!("failed to flush probe output: {error}")))?;
    let output_bytes = output_file
        .metadata()
        .map_err(|error| AppError::output(format!("failed to stat probe output: {error}")))?
        .len();
    let expected_output_bytes =
        (STOP_ORDINAL_EXCLUSIVE - START_ORDINAL) * CANONICAL_RECORD_BYTES as u64;
    if output_bytes != expected_output_bytes {
        return Err(AppError::output(format!(
            "probe output has {output_bytes} bytes instead of {expected_output_bytes}"
        )));
    }

    let digest = hex(&hasher.finalize());
    render_report(
        reader_arguments,
        SourceIdentity {
            mode: source_mode,
            bytes: source_bytes,
        },
        &reader,
        &scan,
        &observed,
        OutputIdentity {
            bytes: output_bytes,
            sha256: &digest,
        },
    )
}

pub(crate) fn validate_authorized_source(
    reader: &E57Reader<BufReader<File>>,
    scans: &[PointCloud],
) -> AppResult<()> {
    if reader.guid() != EXPECTED_FILE_GUID
        || reader.format_name() != "ASTM E57 3D Imaging Data File"
        || scans.len() != EXPECTED_SCANS.len()
    {
        return Err(AppError::validation(
            "E57 root or scan count differs from the frozen source inventory",
        ));
    }
    for (ordinal, (scan, expected)) in scans.iter().zip(EXPECTED_SCANS).enumerate() {
        if scan.guid.as_deref() != Some(expected.0)
            || scan.records != expected.1
            || scan.file_offset != expected.2
        {
            return Err(AppError::validation(format!(
                "scan {ordinal} GUID/count/offset differs from the frozen source inventory"
            )));
        }
        validate_authorized_prototype(scan, ordinal)?;
    }
    let pose = scans[SCAN_ORDINAL]
        .transform
        .as_ref()
        .ok_or_else(|| AppError::validation("authorized scan 0 pose is absent"))?;
    let pose_bits = [
        pose.rotation.w.to_bits(),
        pose.rotation.x.to_bits(),
        pose.rotation.y.to_bits(),
        pose.rotation.z.to_bits(),
        pose.translation.x.to_bits(),
        pose.translation.y.to_bits(),
        pose.translation.z.to_bits(),
    ];
    if pose_bits
        != [
            0x3fefffdb1494d92a,
            0,
            0,
            0xbf784df35c98ab0c,
            0,
            0,
            0x3ff8a3d70a3d70a6,
        ]
    {
        return Err(AppError::validation(
            "scan 0 pose differs from the frozen source inventory",
        ));
    }
    Ok(())
}

fn validate_authorized_prototype(scan: &PointCloud, ordinal: usize) -> AppResult<()> {
    let prototype = &scan.prototype;
    let scaled_xyz = |index: usize, name: RecordName| {
        prototype.get(index).is_some_and(|record| {
            record.name == name
                && matches!(
                    record.data_type,
                    RecordDataType::ScaledInteger {
                        min: -2_147_483_647,
                        max: 2_147_483_647,
                        scale,
                        offset: 0.0,
                    } if scale.to_bits() == 0x3ee4f8b588e368f1
                )
        })
    };
    let exact = scaled_xyz(0, RecordName::CartesianX)
        && scaled_xyz(1, RecordName::CartesianY)
        && scaled_xyz(2, RecordName::CartesianZ)
        && prototype.get(3).is_some_and(|record| {
            record.name == RecordName::Intensity
                && matches!(
                    record.data_type,
                    RecordDataType::Single {
                        min: Some(min),
                        max: Some(max),
                    } if min.to_bits() == 0.0_f32.to_bits() && max.to_bits() == 1.0_f32.to_bits()
                )
        })
        && prototype.get(4).is_some_and(|record| {
            record.name == RecordName::RowIndex
                && matches!(
                    record.data_type,
                    RecordDataType::Integer {
                        min: 0,
                        max: 11_003
                    }
                )
        })
        && prototype.get(5).is_some_and(|record| {
            record.name == RecordName::ColumnIndex
                && matches!(
                    record.data_type,
                    RecordDataType::Integer {
                        min: 0,
                        max: 27_480
                    }
                )
        })
        && prototype.get(6).is_some_and(|record| {
            record.name == RecordName::CartesianInvalidState
                && matches!(record.data_type, RecordDataType::Integer { min: 0, max: 2 })
        });
    if !exact {
        return Err(AppError::validation(format!(
            "scan {ordinal} prototype order or parameters differ from frozen HY_SPRUCE4"
        )));
    }
    Ok(())
}

struct SourceIdentity<'a> {
    mode: &'a str,
    bytes: u64,
}

struct OutputIdentity<'a> {
    bytes: u64,
    sha256: &'a str,
}

fn open_output(descriptor: u32, input_device: u64, input_inode: u64) -> AppResult<File> {
    let path = format!("/dev/fd/{descriptor}");
    let mut file = OpenOptions::new()
        .write(true)
        .open(&path)
        .map_err(|error| {
            AppError::output(format!("failed to duplicate inherited {path}: {error}"))
        })?;
    let metadata = file
        .metadata()
        .map_err(|error| AppError::output(format!("failed to stat probe output: {error}")))?;
    if !metadata.is_file() {
        return Err(AppError::output(
            "inherited probe output descriptor is not a regular file",
        ));
    }
    if metadata.dev() == input_device && metadata.ino() == input_inode {
        return Err(AppError::output(
            "probe output descriptor aliases the E57 input inode",
        ));
    }
    if metadata.len() != 0 {
        return Err(AppError::output(
            "inherited probe output must be a new zero-length file",
        ));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| AppError::output(format!("failed to rewind probe output: {error}")))?;
    Ok(file)
}

pub(crate) fn validate_scan_set(
    scans: &[PointCloud],
    arguments: &DryRunArgs,
    file_bytes: u64,
) -> AppResult<()> {
    if scans.is_empty() || scans.len() > arguments.max_scans {
        return Err(AppError::validation(format!(
            "scan count {} is outside the configured 1..={} range",
            scans.len(),
            arguments.max_scans
        )));
    }
    let mut total_records = 0_u64;
    let mut guids = HashSet::new();
    for (ordinal, scan) in scans.iter().enumerate() {
        if scan.prototype.len() > arguments.max_prototype_fields {
            return Err(AppError::validation(format!(
                "scan {ordinal} prototype exceeds the {}-field ceiling",
                arguments.max_prototype_fields
            )));
        }
        validate_scan(scan, ordinal, file_bytes)?;
        let guid = scan.guid.as_deref().unwrap_or_default();
        if !guids.insert(guid.to_owned()) {
            return Err(AppError::validation(format!(
                "scan {ordinal} repeats GUID {guid}"
            )));
        }
        total_records = total_records
            .checked_add(scan.records)
            .ok_or_else(|| AppError::validation("declared point count overflow"))?;
        if total_records > arguments.max_declared_records {
            return Err(AppError::validation(format!(
                "declared point count exceeds the {}-record ceiling",
                arguments.max_declared_records
            )));
        }
    }
    Ok(())
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

#[derive(Default)]
struct Observed {
    decoded: u64,
    valid_count: u64,
    direction_count: u64,
    invalid_count: u64,
    finite_intensity_count: u64,
    non_finite_intensity_count: u64,
    row: I64Bounds,
    column: I64Bounds,
    raw_x: I64Bounds,
    raw_y: I64Bounds,
    raw_z: I64Bounds,
    local_x: F64Bounds,
    local_y: F64Bounds,
    local_z: F64Bounds,
    valid_x: F64Bounds,
    valid_y: F64Bounds,
    valid_z: F64Bounds,
    valid_range: F64Bounds,
    finite_intensity: F64Bounds,
}

#[derive(Default)]
struct I64Bounds {
    min: Option<i64>,
    max: Option<i64>,
}

impl I64Bounds {
    fn add(&mut self, value: i64) {
        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = Some(self.max.map_or(value, |current| current.max(value)));
    }
}

#[derive(Default)]
struct F64Bounds {
    min: Option<f64>,
    max: Option<f64>,
}

impl F64Bounds {
    fn add(&mut self, value: f64) -> AppResult<()> {
        if !value.is_finite() {
            return Err(AppError::validation(
                "coordinate or derived range is non-finite",
            ));
        }
        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = Some(self.max.map_or(value, |current| current.max(value)));
        Ok(())
    }
}

fn observe_record(
    ordinal: u64,
    values: &RawValues,
    scan: &PointCloud,
    indices: &PrototypeIndices,
    observed: &mut Observed,
) -> AppResult<[u8; CANONICAL_RECORD_BYTES]> {
    let raw_x = scaled_integer(values, indices.x, "cartesianX")?;
    let raw_y = scaled_integer(values, indices.y, "cartesianY")?;
    let raw_z = scaled_integer(values, indices.z, "cartesianZ")?;
    let local_x = values[indices.x]
        .to_f64(&scan.prototype[indices.x].data_type)
        .map_err(|error| AppError::validation(format!("cartesianX conversion failed: {error}")))?;
    let local_y = values[indices.y]
        .to_f64(&scan.prototype[indices.y].data_type)
        .map_err(|error| AppError::validation(format!("cartesianY conversion failed: {error}")))?;
    let local_z = values[indices.z]
        .to_f64(&scan.prototype[indices.z].data_type)
        .map_err(|error| AppError::validation(format!("cartesianZ conversion failed: {error}")))?;
    let row = integer(values, indices.row, "rowIndex")?;
    let column = integer(values, indices.column, "columnIndex")?;
    let invalid = integer(values, indices.invalid, "cartesianInvalidState")?;
    let intensity = match &values[indices.intensity] {
        RecordValue::Single(value) => *value,
        _ => {
            return Err(AppError::validation(
                "intensity raw value does not match the Single prototype",
            ));
        }
    };

    let range = match invalid {
        0 => {
            observed.valid_count += 1;
            observed.valid_x.add(local_x)?;
            observed.valid_y.add(local_y)?;
            observed.valid_z.add(local_z)?;
            let value = (local_x * local_x + local_y * local_y + local_z * local_z).sqrt();
            observed.valid_range.add(value)?;
            Some(value)
        }
        1 => {
            observed.direction_count += 1;
            None
        }
        2 => {
            observed.invalid_count += 1;
            None
        }
        value => {
            return Err(AppError::validation(format!(
                "cartesianInvalidState contains out-of-domain value {value} at ordinal {ordinal}"
            )));
        }
    };

    observed.decoded += 1;
    observed.row.add(row);
    observed.column.add(column);
    observed.raw_x.add(raw_x);
    observed.raw_y.add(raw_y);
    observed.raw_z.add(raw_z);
    observed.local_x.add(local_x)?;
    observed.local_y.add(local_y)?;
    observed.local_z.add(local_z)?;
    if intensity.is_finite() {
        observed.finite_intensity_count += 1;
        observed.finite_intensity.add(f64::from(intensity))?;
    } else {
        observed.non_finite_intensity_count += 1;
    }

    let flags = u32::from(range.is_some()) | (u32::from(intensity.is_finite()) << 1);
    let mut record = [0_u8; CANONICAL_RECORD_BYTES];
    let mut offset = 0;
    append_bytes(&mut record, &mut offset, &ordinal.to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &row.to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &column.to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &invalid.to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &raw_x.to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &raw_y.to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &raw_z.to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &local_x.to_bits().to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &local_y.to_bits().to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &local_z.to_bits().to_le_bytes())?;
    append_bytes(
        &mut record,
        &mut offset,
        &range.unwrap_or(0.0).to_bits().to_le_bytes(),
    )?;
    append_bytes(&mut record, &mut offset, &intensity.to_bits().to_le_bytes())?;
    append_bytes(&mut record, &mut offset, &flags.to_le_bytes())?;
    if offset != CANONICAL_RECORD_BYTES {
        return Err(AppError::output(
            "canonical probe record layout does not total 96 bytes",
        ));
    }
    Ok(record)
}

fn append_bytes(
    output: &mut [u8; CANONICAL_RECORD_BYTES],
    offset: &mut usize,
    value: &[u8],
) -> AppResult<()> {
    let end = offset
        .checked_add(value.len())
        .ok_or_else(|| AppError::output("canonical probe record offset overflow"))?;
    let destination = output
        .get_mut(*offset..end)
        .ok_or_else(|| AppError::output("canonical probe record exceeds 96 bytes"))?;
    destination.copy_from_slice(value);
    *offset = end;
    Ok(())
}

fn scaled_integer(values: &RawValues, index: usize, label: &str) -> AppResult<i64> {
    match &values[index] {
        RecordValue::ScaledInteger(value) => Ok(*value),
        _ => Err(AppError::validation(format!(
            "{label} raw value does not match its ScaledInteger prototype"
        ))),
    }
}

fn integer(values: &RawValues, index: usize, label: &str) -> AppResult<i64> {
    match &values[index] {
        RecordValue::Integer(value) => Ok(*value),
        _ => Err(AppError::validation(format!(
            "{label} raw value does not match its Integer prototype"
        ))),
    }
}

fn render_report(
    arguments: &DryRunArgs,
    source: SourceIdentity<'_>,
    reader: &E57Reader<BufReader<std::fs::File>>,
    scan: &PointCloud,
    observed: &Observed,
    retained: OutputIdentity<'_>,
) -> AppResult<String> {
    let mut output = String::with_capacity(16 * 1024);
    output.push_str("{\"schemaVersion\":");
    push_string(&mut output, PROBE_SCHEMA);
    output.push_str(
        ",\"status\":\"bounded_probe_complete\",\"profile\":\"hy-spruce4\",\"authority\":{",
    );
    write!(
        output,
        "\"scanOrdinal\":{SCAN_ORDINAL},\"scanGuid\":\"{EXPECTED_GUID}\",\"startOrdinalInclusive\":\"{START_ORDINAL}\",\"stopOrdinalExclusive\":\"{STOP_ORDINAL_EXCLUSIVE}\""
    )
    .map_err(|_| AppError::output("failed to format probe authority"))?;
    output.push_str("},\"reader\":{\"adapterVersion\":");
    push_string(&mut output, env!("CARGO_PKG_VERSION"));
    output.push_str(",\"upstreamE57Version\":");
    push_string(&mut output, UPSTREAM_E57_VERSION);
    output.push_str(",\"upstreamCrateSha256\":");
    push_string(&mut output, UPSTREAM_E57_CRATE_SHA256);
    output.push_str(",\"vendorPatchSetSha256\":");
    push_string(&mut output, VENDOR_PATCH_SET_SHA256);
    output.push_str(",\"reusableRawBuffer\":true,\"crcImplementation\":\"vendored_e57_pure_rust\",\"crc32cFeature\":false},\"source\":{");
    write!(
        output,
        "\"inputMode\":\"{}\",\"bytes\":\"{}\",\"digestComputed\":false,\"fileGuid\":",
        source.mode, source.bytes,
    )
    .map_err(|_| AppError::output("failed to format probe source"))?;
    push_string(&mut output, reader.guid());
    output.push_str("},\"scan\":{\"ordinal\":0,\"guid\":");
    push_string(&mut output, scan.guid.as_deref().unwrap_or_default());
    write!(
        output,
        ",\"publisherDeclaredRecords\":\"{}\",\"poseApplied\":false,\"poseProvenanceOnly\":true,\"pose\":",
        scan.records,
    )
    .map_err(|_| AppError::output("failed to format probe scan"))?;
    append_pose(&mut output, scan)?;
    output.push_str(",\"observedPrototype\":[");
    for (index, record) in scan.prototype.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(output, "{{\"index\":{index},\"name\":")
            .map_err(|_| AppError::output("failed to format prototype index"))?;
        push_string(&mut output, &record_name(record));
        output.push_str(",\"kind\":");
        push_string(&mut output, data_type_name(&record.data_type));
        output.push('}');
    }
    output.push_str("]},\"canonicalRecordLayout\":{\"byteOrder\":\"little_endian\",\"bytesPerRecord\":96,\"fields\":[\"sourceOrdinal:u64\",\"row:i64\",\"column:i64\",\"cartesianInvalidState:i64\",\"rawX:i64\",\"rawY:i64\",\"rawZ:i64\",\"localX:f64bits\",\"localY:f64bits\",\"localZ:f64bits\",\"derivedRangeOrZero:f64bits\",\"rawIntensitySingleBits:u32\",\"flags:u32\"],\"flags\":{\"bit0\":\"range_present\",\"bit1\":\"intensity_finite\"}},\"output\":{");
    write!(
        output,
        "\"mode\":\"inherited_fd\",\"records\":\"{}\",\"bytes\":\"{}\",\"sha256\":\"{}\"}},\"result\":{{",
        observed.decoded, retained.bytes, retained.sha256,
    )
    .map_err(|_| AppError::output("failed to format probe output identity"))?;
    write!(
        output,
        "\"decodedRecords\":\"{}\",\"canonicalRecordBytesHashed\":\"{}\",\"canonicalRecordStreamSha256\":\"{}\",\"validCount\":\"{}\",\"directionCount\":\"{}\",\"invalidCount\":\"{}\",\"finiteIntensityCount\":\"{}\",\"nonFiniteIntensityCount\":\"{}\",",
        observed.decoded,
        observed.decoded * CANONICAL_RECORD_BYTES as u64,
        retained.sha256,
        observed.valid_count,
        observed.direction_count,
        observed.invalid_count,
        observed.finite_intensity_count,
        observed.non_finite_intensity_count,
    )
    .map_err(|_| AppError::output("failed to format probe counters"))?;
    output.push_str("\"rowBounds\":");
    append_i64_bounds(&mut output, &observed.row)?;
    output.push_str(",\"columnBounds\":");
    append_i64_bounds(&mut output, &observed.column)?;
    output.push_str(",\"rawXBounds\":");
    append_i64_bounds(&mut output, &observed.raw_x)?;
    output.push_str(",\"rawYBounds\":");
    append_i64_bounds(&mut output, &observed.raw_y)?;
    output.push_str(",\"rawZBounds\":");
    append_i64_bounds(&mut output, &observed.raw_z)?;
    output.push_str(",\"localXBounds\":");
    append_f64_bounds(&mut output, &observed.local_x)?;
    output.push_str(",\"localYBounds\":");
    append_f64_bounds(&mut output, &observed.local_y)?;
    output.push_str(",\"localZBounds\":");
    append_f64_bounds(&mut output, &observed.local_z)?;
    output.push_str(",\"validLocalXBounds\":");
    append_f64_bounds(&mut output, &observed.valid_x)?;
    output.push_str(",\"validLocalYBounds\":");
    append_f64_bounds(&mut output, &observed.valid_y)?;
    output.push_str(",\"validLocalZBounds\":");
    append_f64_bounds(&mut output, &observed.valid_z)?;
    output.push_str(",\"validRangeBoundsM\":");
    append_f64_bounds(&mut output, &observed.valid_range)?;
    output.push_str(",\"finiteIntensityBounds\":");
    append_f64_bounds(&mut output, &observed.finite_intensity)?;
    output.push_str("},\"eofReached\":false,\"publisherCountValidated\":false,\"surfaceClaim\":false,\"recordsRetained\":true,\"recordOutputPublished\":false,\"invalidRecordsRetained\":true,\"poseApplied\":false,\"boundedProbeImplemented\":true,\"fullExtractionImplemented\":false,\"colorAvailability\":\"absent_in_prototype\",\"limits\":{");
    write!(
        output,
        "\"maxFileBytes\":\"{}\",\"maxXmlBytes\":\"{}\",\"maxScans\":{},\"maxPrototypeFields\":{},\"maxDeclaredRecords\":\"{}\",\"maxJsonBytes\":{}",
        arguments.max_file_bytes,
        arguments.max_xml_bytes,
        arguments.max_scans,
        arguments.max_prototype_fields,
        arguments.max_declared_records,
        arguments.max_json_bytes,
    )
    .map_err(|_| AppError::output("failed to format probe limits"))?;
    output.push_str("}}\n");
    ensure_limit(&output, arguments.max_json_bytes)?;
    Ok(output)
}

fn append_pose(output: &mut String, scan: &PointCloud) -> AppResult<()> {
    let transform = scan
        .transform
        .as_ref()
        .ok_or_else(|| AppError::validation("validated scan unexpectedly lost its pose"))?;
    let q = &transform.rotation;
    let t = &transform.translation;
    write!(
        output,
        "{{\"rotation\":{{\"w\":{},\"x\":{},\"y\":{},\"z\":{},\"wBits\":\"{:016x}\",\"xBits\":\"{:016x}\",\"yBits\":\"{:016x}\",\"zBits\":\"{:016x}\"}},\"translation\":{{\"x\":{},\"y\":{},\"z\":{},\"xBits\":\"{:016x}\",\"yBits\":\"{:016x}\",\"zBits\":\"{:016x}\"}}}}",
        q.w,
        q.x,
        q.y,
        q.z,
        q.w.to_bits(),
        q.x.to_bits(),
        q.y.to_bits(),
        q.z.to_bits(),
        t.x,
        t.y,
        t.z,
        t.x.to_bits(),
        t.y.to_bits(),
        t.z.to_bits(),
    )
    .map_err(|_| AppError::output("failed to format probe pose"))
}

fn data_type_name(data_type: &RecordDataType) -> &'static str {
    match data_type {
        RecordDataType::Single { .. } => "Single",
        RecordDataType::Double { .. } => "Double",
        RecordDataType::ScaledInteger { .. } => "ScaledInteger",
        RecordDataType::Integer { .. } => "Integer",
    }
}

fn append_i64_bounds(output: &mut String, bounds: &I64Bounds) -> AppResult<()> {
    match (bounds.min, bounds.max) {
        (Some(min), Some(max)) => write!(output, "{{\"minimum\":\"{min}\",\"maximum\":\"{max}\"}}")
            .map_err(|_| AppError::output("failed to format integer bounds")),
        _ => {
            output.push_str("null");
            Ok(())
        }
    }
}

fn append_f64_bounds(output: &mut String, bounds: &F64Bounds) -> AppResult<()> {
    match (bounds.min, bounds.max) {
        (Some(min), Some(max)) => write!(output, "{{\"minimum\":{min},\"maximum\":{max}}}")
            .map_err(|_| AppError::output("failed to format float bounds")),
        _ => {
            output.push_str("null");
            Ok(())
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}
