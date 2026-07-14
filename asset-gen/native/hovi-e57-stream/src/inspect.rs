use crate::args::DryRunArgs;
use crate::error::{AppError, AppResult};
use crate::input::open_and_preflight;
use crate::json::{ensure_limit, push_string};
use crate::provenance::{
    DRY_RUN_SCHEMA, UPSTREAM_E57_CRATE_SHA256, UPSTREAM_E57_VCS_COMMIT, UPSTREAM_E57_VERSION,
    VENDOR_PATCH_SET_SHA256,
};
use crate::schema::{record_name, validate_scan};
use e57::{E57Reader, PointCloud, Record, RecordDataType};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::BufReader;

pub fn dry_run(arguments: &DryRunArgs) -> AppResult<String> {
    let input = open_and_preflight(
        &arguments.input,
        arguments.max_file_bytes,
        arguments.max_xml_bytes,
    )?;
    let source_mode = input.mode;
    let source_bytes = input.bytes;
    let source_header = input.header;
    let reader = E57Reader::new(BufReader::with_capacity(1 << 20, input.file))
        .map_err(|error| AppError::validation(format!("E57 metadata parse failed: {error}")))?;
    let scans = reader.pointclouds();
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
        validate_scan(scan, ordinal, source_bytes)?;
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

    let mut output = String::with_capacity(64 * 1024);
    output.push_str("{\"schemaVersion\":");
    push_string(&mut output, DRY_RUN_SCHEMA);
    output.push_str(",\"status\":\"metadata_validated_no_records_decoded\",\"profile\":\"hy-spruce4\",\"reader\":{");
    output.push_str("\"adapterVersion\":");
    push_string(&mut output, env!("CARGO_PKG_VERSION"));
    output.push_str(",\"upstreamE57Version\":");
    push_string(&mut output, UPSTREAM_E57_VERSION);
    output.push_str(",\"upstreamCrateSha256\":");
    push_string(&mut output, UPSTREAM_E57_CRATE_SHA256);
    output.push_str(",\"upstreamVcsCommit\":");
    push_string(&mut output, UPSTREAM_E57_VCS_COMMIT);
    output.push_str(",\"vendorPatchSetSha256\":");
    push_string(&mut output, VENDOR_PATCH_SET_SHA256);
    output.push_str(",\"crcImplementation\":\"vendored_e57_pure_rust\",\"crc32cFeature\":false,\"reusableRawBuffer\":true,\"boundedProbeImplemented\":true,\"fullExtractionImplemented\":false},\"source\":{");
    write!(
        output,
        "\"inputMode\":\"{}\",\"bytes\":\"{}\",\"digestComputed\":false,\"header\":{{\"major\":{},\"minor\":{},\"physicalLength\":\"{}\",\"xmlOffset\":\"{}\",\"xmlLength\":\"{}\",\"pageSize\":\"{}\"}}",
        source_mode,
        source_bytes,
        source_header.major,
        source_header.minor,
        source_header.physical_length,
        source_header.xml_offset,
        source_header.xml_length,
        source_header.page_size,
    )
    .map_err(|_| AppError::output("failed to format source metadata"))?;
    output.push_str("},\"e57\":{\"fileGuid\":");
    push_string(&mut output, reader.guid());
    output.push_str(",\"formatName\":");
    push_string(&mut output, reader.format_name());
    output.push_str(",\"libraryVersion\":");
    if let Some(version) = reader.library_version() {
        push_string(&mut output, version);
    } else {
        output.push_str("null");
    }
    write!(
        output,
        ",\"xmlBytes\":\"{}\",\"scanCount\":{},\"declaredRecordCount\":\"{}\",\"scans\":[",
        reader.xml().len(),
        scans.len(),
        total_records,
    )
    .map_err(|_| AppError::output("failed to format E57 summary"))?;

    for (ordinal, scan) in scans.iter().enumerate() {
        if ordinal > 0 {
            output.push(',');
        }
        append_scan(&mut output, scan, ordinal)?;
        ensure_limit(&output, arguments.max_json_bytes)?;
    }
    output.push_str("]},\"limits\":{");
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
    .map_err(|_| AppError::output("failed to format resource limits"))?;
    output.push_str("},\"compressedVectorRecordsDecoded\":\"0\",\"pointPayloadRead\":false,\"colorAvailability\":\"absent_in_prototype\"}\n");
    ensure_limit(&output, arguments.max_json_bytes)?;
    Ok(output)
}

fn append_scan(output: &mut String, scan: &PointCloud, ordinal: usize) -> AppResult<()> {
    output.push_str("{\"ordinal\":");
    write!(output, "{ordinal}").map_err(|_| AppError::output("failed to format scan ordinal"))?;
    output.push_str(",\"guid\":");
    push_string(output, scan.guid.as_deref().unwrap_or_default());
    write!(
        output,
        ",\"records\":\"{}\",\"fileOffset\":\"{}\",\"colorPresent\":false,\"pose\":",
        scan.records, scan.file_offset,
    )
    .map_err(|_| AppError::output("failed to format scan metadata"))?;
    append_pose(output, scan)?;
    output.push_str(",\"prototype\":[");
    for (index, record) in scan.prototype.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        append_record(output, record)?;
    }
    output.push_str("]}");
    Ok(())
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
        "{{\"rotation\":{{\"w\":{},\"x\":{},\"y\":{},\"z\":{}}},\"translation\":{{\"x\":{},\"y\":{},\"z\":{}}}}}",
        q.w, q.x, q.y, q.z, t.x, t.y, t.z,
    )
    .map_err(|_| AppError::output("failed to format scan pose"))
}

fn append_record(output: &mut String, record: &Record) -> AppResult<()> {
    output.push_str("{\"name\":");
    push_string(output, &record_name(record));
    output.push_str(",\"dataType\":");
    match &record.data_type {
        RecordDataType::Single { min, max } => {
            output.push_str("{\"kind\":\"Single\",\"minimum\":");
            append_optional_float(output, min.map(f64::from))?;
            output.push_str(",\"maximum\":");
            append_optional_float(output, max.map(f64::from))?;
            output.push('}');
        }
        RecordDataType::Double { min, max } => {
            output.push_str("{\"kind\":\"Double\",\"minimum\":");
            append_optional_float(output, *min)?;
            output.push_str(",\"maximum\":");
            append_optional_float(output, *max)?;
            output.push('}');
        }
        RecordDataType::ScaledInteger {
            min,
            max,
            scale,
            offset,
        } => {
            write!(
                output,
                "{{\"kind\":\"ScaledInteger\",\"minimum\":\"{min}\",\"maximum\":\"{max}\",\"scale\":{scale},\"offset\":{offset}}}"
            )
            .map_err(|_| AppError::output("failed to format scaled integer type"))?;
        }
        RecordDataType::Integer { min, max } => {
            write!(
                output,
                "{{\"kind\":\"Integer\",\"minimum\":\"{min}\",\"maximum\":\"{max}\"}}"
            )
            .map_err(|_| AppError::output("failed to format integer type"))?;
        }
    }
    output.push('}');
    Ok(())
}

fn append_optional_float(output: &mut String, value: Option<f64>) -> AppResult<()> {
    if let Some(value) = value {
        if !value.is_finite() {
            return Err(AppError::validation(
                "prototype contains a non-finite float limit",
            ));
        }
        write!(output, "{value}").map_err(|_| AppError::output("failed to format float limit"))?;
    } else {
        output.push_str("null");
    }
    Ok(())
}
