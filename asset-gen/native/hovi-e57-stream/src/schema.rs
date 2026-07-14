use crate::error::{AppError, AppResult};
use e57::{PointCloud, Record, RecordDataType, RecordName};
use std::collections::HashSet;

const EXPECTED_FIELDS: [&str; 7] = [
    "cartesianX",
    "cartesianY",
    "cartesianZ",
    "intensity",
    "rowIndex",
    "columnIndex",
    "cartesianInvalidState",
];

pub fn validate_scan(scan: &PointCloud, ordinal: usize, file_bytes: u64) -> AppResult<()> {
    let guid = scan
        .guid
        .as_deref()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AppError::validation(format!("scan {ordinal} has no non-empty GUID")))?;
    if scan.records == 0 {
        return Err(AppError::validation(format!(
            "scan {ordinal} ({guid}) declares zero records"
        )));
    }
    if scan.file_offset == 0 || scan.file_offset >= file_bytes {
        return Err(AppError::validation(format!(
            "scan {ordinal} ({guid}) compressed-vector offset is outside the file"
        )));
    }
    if scan.prototype.len() != EXPECTED_FIELDS.len() {
        return Err(AppError::validation(format!(
            "scan {ordinal} ({guid}) prototype has {} fields instead of {}",
            scan.prototype.len(),
            EXPECTED_FIELDS.len()
        )));
    }

    let mut names = HashSet::new();
    for record in &scan.prototype {
        let name = record_name(record);
        if !names.insert(name.clone()) {
            return Err(AppError::validation(format!(
                "scan {ordinal} ({guid}) has duplicate prototype field {name}"
            )));
        }
        validate_record_type(record, ordinal, guid)?;
    }
    for expected in EXPECTED_FIELDS {
        if !names.contains(expected) {
            return Err(AppError::validation(format!(
                "scan {ordinal} ({guid}) is missing required prototype field {expected}"
            )));
        }
    }
    if scan.has_color() {
        return Err(AppError::validation(format!(
            "scan {ordinal} ({guid}) unexpectedly has color fields"
        )));
    }
    validate_pose(scan, ordinal, guid)?;
    Ok(())
}

pub fn record_name(record: &Record) -> String {
    match &record.name {
        RecordName::CartesianX => "cartesianX".to_owned(),
        RecordName::CartesianY => "cartesianY".to_owned(),
        RecordName::CartesianZ => "cartesianZ".to_owned(),
        RecordName::CartesianInvalidState => "cartesianInvalidState".to_owned(),
        RecordName::SphericalRange => "sphericalRange".to_owned(),
        RecordName::SphericalAzimuth => "sphericalAzimuth".to_owned(),
        RecordName::SphericalElevation => "sphericalElevation".to_owned(),
        RecordName::SphericalInvalidState => "sphericalInvalidState".to_owned(),
        RecordName::Intensity => "intensity".to_owned(),
        RecordName::IsIntensityInvalid => "isIntensityInvalid".to_owned(),
        RecordName::ColorRed => "colorRed".to_owned(),
        RecordName::ColorGreen => "colorGreen".to_owned(),
        RecordName::ColorBlue => "colorBlue".to_owned(),
        RecordName::IsColorInvalid => "isColorInvalid".to_owned(),
        RecordName::RowIndex => "rowIndex".to_owned(),
        RecordName::ColumnIndex => "columnIndex".to_owned(),
        RecordName::ReturnCount => "returnCount".to_owned(),
        RecordName::ReturnIndex => "returnIndex".to_owned(),
        RecordName::TimeStamp => "timeStamp".to_owned(),
        RecordName::IsTimeStampInvalid => "isTimeStampInvalid".to_owned(),
        RecordName::Unknown { namespace, name } => format!("{namespace}:{name}"),
    }
}

fn validate_record_type(record: &Record, ordinal: usize, guid: &str) -> AppResult<()> {
    let name = record_name(record);
    let valid = match &record.name {
        RecordName::CartesianX | RecordName::CartesianY | RecordName::CartesianZ => {
            matches!(&record.data_type, RecordDataType::ScaledInteger { .. })
        }
        RecordName::Intensity => matches!(&record.data_type, RecordDataType::Single { .. }),
        RecordName::RowIndex | RecordName::ColumnIndex | RecordName::CartesianInvalidState => {
            matches!(&record.data_type, RecordDataType::Integer { .. })
        }
        _ => false,
    };
    if !valid {
        return Err(AppError::validation(format!(
            "scan {ordinal} ({guid}) field {name} has an unsupported type for HY_SPRUCE4"
        )));
    }
    Ok(())
}

fn validate_pose(scan: &PointCloud, ordinal: usize, guid: &str) -> AppResult<()> {
    let transform = scan.transform.as_ref().ok_or_else(|| {
        AppError::validation(format!("scan {ordinal} ({guid}) has no pose transform"))
    })?;
    let quaternion = &transform.rotation;
    let translation = &transform.translation;
    let values = [
        quaternion.w,
        quaternion.x,
        quaternion.y,
        quaternion.z,
        translation.x,
        translation.y,
        translation.z,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(AppError::validation(format!(
            "scan {ordinal} ({guid}) pose contains a non-finite component"
        )));
    }
    if quaternion.w < 0.0 {
        return Err(AppError::validation(format!(
            "scan {ordinal} ({guid}) quaternion scalar is negative"
        )));
    }
    let norm = (quaternion.w * quaternion.w
        + quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z)
        .sqrt();
    if (norm - 1.0).abs() > 1.0e-6 {
        return Err(AppError::validation(format!(
            "scan {ordinal} ({guid}) quaternion norm {norm} is not unit length"
        )));
    }
    Ok(())
}
