use crate::bs_write::ByteStreamWriteBuffer;
use crate::error::Converter;
use crate::{Error, Result};
use roxmltree::Node;
use std::error::Error as StdError;
use std::fmt::{Debug, Display};
use std::str::FromStr;

const ASTM_E57_NAMESPACE: &str = "http://www.astm.org/COMMIT/E57/2010-e57-v1.0";

/// Describes a record inside a E57 file with name and data type.
#[derive(Clone, Debug)]
pub struct Record {
    pub name: RecordName,
    pub data_type: RecordDataType,
}

/// Basic primitive E57 data types that are used for the different point attributes.
#[derive(Clone, Debug)]
pub enum RecordDataType {
    /// 32-bit IEEE 754-2008 floating point value.
    Single { min: Option<f32>, max: Option<f32> },
    /// 64-bit IEEE 754-2008 floating point value.
    Double { min: Option<f64>, max: Option<f64> },
    /// Signed 64-bit integer scaled with a fixed 64-bit floating point value.
    ScaledInteger {
        min: i64,
        max: i64,
        scale: f64,
        offset: f64,
    },
    /// Signed 64-bit integer value.
    Integer { min: i64, max: i64 },
}

/// Used to describe the prototype records with all attributes that exist in the point cloud.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum RecordName {
    /// Cartesian X coordinate (in meters).
    CartesianX,
    /// Cartesian Y coordinate (in meters).
    CartesianY,
    /// Cartesian Z coordinate (in meters).
    CartesianZ,
    /// Indicates whether the Cartesian coordinate or its magnitude is meaningful.
    /// Can have the value 0 (valid), 1 (XYZ is a direction vector) or 2 (invalid).
    CartesianInvalidState,

    /// Non-negative range (in meters) of the spherical coordinate.
    SphericalRange,
    /// Azimuth angle (in radians between -PI and PI) of the spherical coordinate.
    SphericalAzimuth,
    // Elevation angle (in radians between -PI/2 and PI/2) of the spherical coordinate.
    SphericalElevation,
    /// Indicates whether the spherical coordinate or its range is meaningful.
    /// Can have the value 0 (valid), 1 (range is not meaningful) or 2 (invalid).
    SphericalInvalidState,

    /// Point intensity. Unit is not specified.
    Intensity,
    /// Indicates whether the intensity value is meaningful.
    /// Can have the value 0 (valid) or 1 (invalid).
    IsIntensityInvalid,

    /// Red color value. Unit is not specified.
    ColorRed,
    /// Green color value. Unit is not specified.
    ColorGreen,
    /// Blue color value. Unit is not specified.
    ColorBlue,
    /// Indicates whether the color value is meaningful.
    /// Can have the value 0 (valid) or 1 (invalid).
    IsColorInvalid,

    /// Row number of the point (zero-based). Used for data that is stored in a grid.
    RowIndex,
    /// Column number of the point (zero-based). Used for data that is stored in a grid.
    ColumnIndex,

    /// For multi-return sensors. The total number of returns for the pulse that this point corresponds to.
    ReturnCount,
    /// For multi-return sensors. The number of this return (zero based). That is, 0 is the first, 1 is the second return etc.
    ReturnIndex,

    /// Non-negative time (in seconds) since the start time given by acquisition start in the parent point cloud.
    TimeStamp,
    /// Indicates whether the time stamp value is meaningful.
    /// Can have the value 0 (valid) or 1 (invalid).
    IsTimeStampInvalid,

    /// Unknown point attribute that is not part of the E57 standard.
    /// Files with such attributes are still valid, since any E57 reader must be able to handle unknown extensions.
    /// Most extensions are described on <http://www.libe57.org/extensions.html>, but others might be proprietary.
    Unknown {
        /// XML namespace of the extension that defines this attribute.
        /// Required, but some E57 files out there are omitting it and we still want to be able to read them.
        /// In such cases the namespace will be set to an empty string.
        namespace: String,
        /// Name of the point atribute.
        name: String,
    },
}

/// Represents a raw value of attributes inside a point cloud.
///
/// For scaled integers the record data type with the scale and offset is needed to calculate the actual f64 value.
#[derive(Clone, Debug, PartialEq)]
pub enum RecordValue {
    Single(f32),
    Double(f64),
    ScaledInteger(i64),
    Integer(i64),
}

impl Record {
    pub(crate) fn xml_string(&self) -> String {
        let namespace = self
            .name
            .namespace()
            .map(|n| n.to_owned() + ":")
            .unwrap_or("".to_owned());
        let tag_name = self.name.tag_name();
        let (attrs, value) = serialize_record_type(&self.data_type);
        format!("<{namespace}{tag_name} {attrs}>{value}</{namespace}{tag_name}>\n")
    }
}

impl RecordName {
    pub(crate) fn tag_name(&self) -> &str {
        match self {
            RecordName::CartesianX => "cartesianX",
            RecordName::CartesianY => "cartesianY",
            RecordName::CartesianZ => "cartesianZ",
            RecordName::CartesianInvalidState => "cartesianInvalidState",
            RecordName::SphericalRange => "sphericalRange",
            RecordName::SphericalAzimuth => "sphericalAzimuth",
            RecordName::SphericalElevation => "sphericalElevation",
            RecordName::SphericalInvalidState => "sphericalInvalidState",
            RecordName::Intensity => "intensity",
            RecordName::IsIntensityInvalid => "isIntensityInvalid",
            RecordName::ColorRed => "colorRed",
            RecordName::ColorGreen => "colorGreen",
            RecordName::ColorBlue => "colorBlue",
            RecordName::IsColorInvalid => "isColorInvalid",
            RecordName::RowIndex => "rowIndex",
            RecordName::ColumnIndex => "columnIndex",
            RecordName::ReturnCount => "returnCount",
            RecordName::ReturnIndex => "returnIndex",
            RecordName::TimeStamp => "timeStamp",
            RecordName::IsTimeStampInvalid => "isTimeStampInvalid",
            RecordName::Unknown { name, .. } => name,
        }
    }

    pub(crate) fn namespace(&self) -> Option<&str> {
        match self {
            RecordName::Unknown { namespace, .. } => Some(namespace),
            _ => None,
        }
    }

    pub(crate) fn from_namespace_and_tag_name(
        namespace_prefix: Option<&str>,
        namespace_uri: Option<&str>,
        tag_name: &str,
    ) -> Result<Self> {
        if namespace_uri != Some(ASTM_E57_NAMESPACE) {
            return Ok(RecordName::Unknown {
                namespace: namespace_prefix.unwrap_or_default().to_owned(),
                name: tag_name.to_owned(),
            });
        }
        Ok(match tag_name {
            "cartesianX" => RecordName::CartesianX,
            "cartesianY" => RecordName::CartesianY,
            "cartesianZ" => RecordName::CartesianZ,
            "cartesianInvalidState" => RecordName::CartesianInvalidState,
            "sphericalRange" => RecordName::SphericalRange,
            "sphericalAzimuth" => RecordName::SphericalAzimuth,
            "sphericalElevation" => RecordName::SphericalElevation,
            "sphericalInvalidState" => RecordName::SphericalInvalidState,
            "intensity" => RecordName::Intensity,
            "isIntensityInvalid" => RecordName::IsIntensityInvalid,
            "colorRed" => RecordName::ColorRed,
            "colorGreen" => RecordName::ColorGreen,
            "colorBlue" => RecordName::ColorBlue,
            "isColorInvalid" => RecordName::IsColorInvalid,
            "rowIndex" => RecordName::RowIndex,
            "columnIndex" => RecordName::ColumnIndex,
            "returnCount" => RecordName::ReturnCount,
            "returnIndex" => RecordName::ReturnIndex,
            "timeStamp" => RecordName::TimeStamp,
            "isTimeStampInvalid" => RecordName::IsTimeStampInvalid,
            _ => RecordName::Unknown {
                namespace: namespace_prefix.unwrap_or_default().to_owned(),
                name: tag_name.to_owned(),
            },
        })
    }
}

impl RecordDataType {
    pub(crate) fn from_node(node: &Node) -> Result<Self> {
        let tag_name = node.tag_name().name();
        let type_name = node
            .attribute("type")
            .invalid_err(format!("Missing type attribute for XML tag '{tag_name}'"))?;
        Ok(match type_name {
            "Float" => {
                let precision = node.attribute("precision").unwrap_or("double");
                if precision == "double" {
                    let min = optional_attribute(node, "minimum", tag_name, type_name)?;
                    let max = optional_attribute(node, "maximum", tag_name, type_name)?;
                    validate_f64_limits(min, max, tag_name)?;
                    RecordDataType::Double { min, max }
                } else if precision == "single" {
                    let min = optional_attribute(node, "minimum", tag_name, type_name)?;
                    let max = optional_attribute(node, "maximum", tag_name, type_name)?;
                    validate_f32_limits(min, max, tag_name)?;
                    RecordDataType::Single { min, max }
                } else {
                    Error::invalid(format!(
                        "Float 'precision' attribute value '{precision}' for 'Float' type is unknown"
                    ))?
                }
            }
            "Integer" => {
                let min =
                    optional_attribute(node, "minimum", tag_name, type_name)?.unwrap_or(i64::MIN);
                let max =
                    optional_attribute(node, "maximum", tag_name, type_name)?.unwrap_or(i64::MAX);
                if max < min {
                    Error::invalid(format!(
                        "Maximum value '{max}' and minimum value '{min}' of type '{type_name}' in XML tag '{tag_name}' are invalid"
                    ))?
                }
                RecordDataType::Integer { min, max }
            }
            "ScaledInteger" => {
                let min =
                    optional_attribute(node, "minimum", tag_name, type_name)?.unwrap_or(i64::MIN);
                let max =
                    optional_attribute(node, "maximum", tag_name, type_name)?.unwrap_or(i64::MAX);
                if max < min {
                    Error::invalid(format!(
                        "Maximum value '{max}' and minimum value '{min}' of type '{type_name}' in XML tag '{tag_name}' are invalid"
                    ))?
                }
                let scale: f64 =
                    optional_attribute(node, "scale", tag_name, type_name)?.unwrap_or(1.0);
                let offset: f64 =
                    optional_attribute(node, "offset", tag_name, type_name)?.unwrap_or(0.0);
                if !scale.is_finite() || scale <= 0.0 {
                    Error::invalid(format!(
                        "Scaled integer scale '{scale}' in XML tag '{tag_name}' must be finite and positive"
                    ))?
                }
                if !offset.is_finite() {
                    Error::invalid(format!(
                        "Scaled integer offset '{offset}' in XML tag '{tag_name}' must be finite"
                    ))?
                }
                RecordDataType::ScaledInteger {
                    min,
                    max,
                    scale,
                    offset,
                }
            }
            _ => Error::not_implemented(format!(
                "Unsupported type '{type_name}' in XML tag '{tag_name}' detected"
            ))?,
        })
    }

    pub(crate) fn bit_size(&self) -> usize {
        match self {
            RecordDataType::Single { .. } => std::mem::size_of::<f32>() * 8,
            RecordDataType::Double { .. } => std::mem::size_of::<f64>() * 8,
            RecordDataType::ScaledInteger { min, max, .. } => integer_bits(*min, *max),
            RecordDataType::Integer { min, max } => integer_bits(*min, *max),
        }
    }

    pub(crate) fn write(
        &self,
        value: &RecordValue,
        buffer: &mut ByteStreamWriteBuffer,
    ) -> Result<()> {
        match self {
            RecordDataType::Single { .. } => {
                if let RecordValue::Single(float) = value {
                    let bytes = float.to_le_bytes();
                    buffer.add_bytes(&bytes);
                } else {
                    Error::invalid("Data type single only supports single values")?
                }
            }
            RecordDataType::Double { .. } => {
                if let RecordValue::Double(double) = value {
                    let bytes = double.to_le_bytes();
                    buffer.add_bytes(&bytes);
                } else {
                    Error::invalid("Data type double only supports double values")?
                }
            }
            RecordDataType::ScaledInteger { min, max, .. } => {
                if let RecordValue::ScaledInteger(int) = value {
                    serialize_integer(*int, *min, *max, buffer);
                } else {
                    Error::invalid("Data type scaled integer only supports scaled integer values")?
                }
            }
            RecordDataType::Integer { min, max } => {
                if let RecordValue::Integer(int) = value {
                    serialize_integer(*int, *min, *max, buffer);
                } else {
                    Error::invalid("Data type integer only supports integer values")?
                }
            }
        };
        Ok(())
    }

    pub(crate) fn limits(&self) -> (Option<RecordValue>, Option<RecordValue>) {
        match self {
            RecordDataType::Single { min, max } => {
                (min.map(RecordValue::Single), max.map(RecordValue::Single))
            }
            RecordDataType::Double { min, max } => {
                (min.map(RecordValue::Double), max.map(RecordValue::Double))
            }
            RecordDataType::ScaledInteger { min, max, .. } => (
                Some(RecordValue::ScaledInteger(*min)),
                Some(RecordValue::ScaledInteger(*max)),
            ),
            RecordDataType::Integer { min, max } => (
                Some(RecordValue::Integer(*min)),
                Some(RecordValue::Integer(*max)),
            ),
        }
    }
}

fn validate_f32_limits(min: Option<f32>, max: Option<f32>, tag_name: &str) -> Result<()> {
    if min.is_some_and(|value| !value.is_finite()) || max.is_some_and(|value| !value.is_finite()) {
        Error::invalid(format!(
            "Float limits in XML tag '{tag_name}' must be finite"
        ))?
    }
    if matches!((min, max), (Some(min), Some(max)) if max < min) {
        Error::invalid(format!(
            "Float maximum is smaller than minimum in XML tag '{tag_name}'"
        ))?
    }
    Ok(())
}

fn validate_f64_limits(min: Option<f64>, max: Option<f64>, tag_name: &str) -> Result<()> {
    if min.is_some_and(|value| !value.is_finite()) || max.is_some_and(|value| !value.is_finite()) {
        Error::invalid(format!(
            "Float limits in XML tag '{tag_name}' must be finite"
        ))?
    }
    if matches!((min, max), (Some(min), Some(max)) if max < min) {
        Error::invalid(format!(
            "Float maximum is smaller than minimum in XML tag '{tag_name}'"
        ))?
    }
    Ok(())
}

impl RecordValue {
    pub fn to_f64(&self, dt: &RecordDataType) -> Result<f64> {
        match self {
            RecordValue::Single(s) => Ok(*s as f64),
            RecordValue::Double(d) => Ok(*d),
            RecordValue::ScaledInteger(i) => {
                if let RecordDataType::ScaledInteger { scale, offset, .. } = dt {
                    Ok(*i as f64 * *scale + offset)
                } else {
                    Error::internal("Tried to convert scaled integer value with wrong data type")
                }
            }
            RecordValue::Integer(i) => Ok(*i as f64),
        }
    }

    pub fn to_u8(&self, dt: &RecordDataType) -> Result<u8> {
        if let (RecordValue::Integer(i), RecordDataType::Integer { min, max }) = (self, dt) {
            if *min >= 0 && *max <= 255 {
                Ok(*i as u8)
            } else {
                Error::internal("Integer range is too big for u8")
            }
        } else {
            Error::internal("Tried to convert value to u8 with unsupported value or data type")
        }
    }

    pub fn to_i64(&self, dt: &RecordDataType) -> Result<i64> {
        // Does not make much sense to convert float to int,
        // but some files generated by PDAL use floats for invalid states...
        // See also here: https://github.com/cry-inc/e57/issues/18

        if let (RecordValue::Integer(i), RecordDataType::Integer { .. }) = (self, dt) {
            Ok(*i)
        } else if let (
            RecordValue::ScaledInteger(si),
            RecordDataType::ScaledInteger { scale, offset, .. },
        ) = (self, dt)
        {
            Ok((*si as f64 * *scale + offset) as i64)
        } else if let (RecordValue::Single(s), RecordDataType::Single { .. }) = (self, dt) {
            Ok(*s as i64)
        } else if let (RecordValue::Double(d), RecordDataType::Double { .. }) = (self, dt) {
            Ok(*d as i64)
        } else {
            Error::internal("Tried to convert value to i64 with unsupported data type")
        }
    }
}

impl Display for RecordValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecordValue::Single(v) => write!(f, "{v}"),
            RecordValue::Double(v) => write!(f, "{v}"),
            RecordValue::ScaledInteger(v) => write!(f, "{v}"),
            RecordValue::Integer(v) => write!(f, "{v}"),
        }
    }
}

#[inline]
fn serialize_integer(value: i64, min: i64, max: i64, buffer: &mut ByteStreamWriteBuffer) {
    let uint = (value - min) as u64;
    let data = uint.to_le_bytes();
    let bits = integer_bits(min, max);
    buffer.add_bits(&data, bits);
}

#[inline]
fn integer_bits(min: i64, max: i64) -> usize {
    let range = max as i128 - min as i128;
    if range > 0 {
        range.ilog2() as usize + 1
    } else {
        0
    }
}

fn optional_attribute<T>(
    node: &Node,
    attribute: &str,
    tag_name: &str,
    type_name: &str,
) -> Result<Option<T>>
where
    T: FromStr,
    T::Err: StdError + Send + Sync + 'static,
{
    Ok(if let Some(attr) = node.attribute(attribute) {
        let parsed = attr.parse::<T>();
        Some(parsed.invalid_err(format!(
            "Failed to parse attribute '{attribute}' for type '{type_name}' in XML tag '{tag_name}'"
        ))?)
    } else {
        None
    })
}

fn serialize_record_type(rt: &RecordDataType) -> (String, String) {
    match rt {
        RecordDataType::Single { min, max } => {
            let mut str = String::from("type=\"Float\" precision=\"single\"");
            if let Some(min) = min {
                str += &format!(" minimum=\"{min}\"");
            }
            if let Some(max) = max {
                str += &format!(" maximum=\"{max}\"");
            }
            let value = min.unwrap_or(0.0).to_string();
            (str, value)
        }
        RecordDataType::Double { min, max } => {
            let mut str = String::from("type=\"Float\"");
            if let Some(min) = min {
                str += &format!(" minimum=\"{min}\"");
            }
            if let Some(max) = max {
                str += &format!(" maximum=\"{max}\"");
            }
            let value = min.unwrap_or(0.0).to_string();
            (str, value)
        }
        RecordDataType::ScaledInteger { min, max, scale, offset } => (
            format!(
                "type=\"ScaledInteger\" minimum=\"{min}\" maximum=\"{max}\" scale=\"{scale}\" offset=\"{offset}\""
            ),
            min.to_string(),
        ),
        RecordDataType::Integer { min, max } => (
            format!("type=\"Integer\" minimum=\"{min}\" maximum=\"{max}\""),
            min.to_string(),
        ),
    }
}

impl RecordDataType {
    pub const F32: RecordDataType = RecordDataType::Single {
        min: None,
        max: None,
    };

    pub const UNIT_F32: RecordDataType = RecordDataType::Single {
        min: Some(0.0),
        max: Some(1.0),
    };

    pub const F64: RecordDataType = RecordDataType::Double {
        min: None,
        max: None,
    };

    pub const U8: RecordDataType = RecordDataType::Integer {
        min: 0,
        max: u8::MAX as i64,
    };

    pub const U16: RecordDataType = RecordDataType::Integer {
        min: 0,
        max: u16::MAX as i64,
    };
}

impl Record {
    pub const CARTESIAN_X_F32: Record = Record {
        name: RecordName::CartesianX,
        data_type: RecordDataType::F32,
    };

    pub const CARTESIAN_Y_F32: Record = Record {
        name: RecordName::CartesianY,
        data_type: RecordDataType::F32,
    };

    pub const CARTESIAN_Z_F32: Record = Record {
        name: RecordName::CartesianZ,
        data_type: RecordDataType::F32,
    };

    pub const CARTESIAN_X_F64: Record = Record {
        name: RecordName::CartesianX,
        data_type: RecordDataType::F64,
    };

    pub const CARTESIAN_Y_F64: Record = Record {
        name: RecordName::CartesianY,
        data_type: RecordDataType::F64,
    };

    pub const CARTESIAN_Z_F64: Record = Record {
        name: RecordName::CartesianZ,
        data_type: RecordDataType::F64,
    };

    pub const CARTESIAN_INVALID_STATE: Record = Record {
        name: RecordName::CartesianInvalidState,
        data_type: RecordDataType::Integer { min: 0, max: 2 },
    };

    pub const SPHERICAL_AZIMUTH_F32: Record = Record {
        name: RecordName::SphericalAzimuth,
        data_type: RecordDataType::F32,
    };

    pub const SPHERICAL_ELEVATION_F32: Record = Record {
        name: RecordName::SphericalElevation,
        data_type: RecordDataType::F32,
    };

    pub const SPHERICAL_RANGE_F32: Record = Record {
        name: RecordName::SphericalRange,
        data_type: RecordDataType::F32,
    };

    pub const SPHERICAL_AZIMUTH_F64: Record = Record {
        name: RecordName::SphericalAzimuth,
        data_type: RecordDataType::F64,
    };

    pub const SPHERICAL_ELEVATION_F64: Record = Record {
        name: RecordName::SphericalElevation,
        data_type: RecordDataType::F64,
    };

    pub const SPHERICAL_RANGE_F64: Record = Record {
        name: RecordName::SphericalRange,
        data_type: RecordDataType::F64,
    };

    pub const SPHERICAL_INVALID_STATE: Record = Record {
        name: RecordName::SphericalInvalidState,
        data_type: RecordDataType::Integer { min: 0, max: 2 },
    };

    pub const COLOR_RED_U8: Record = Record {
        name: RecordName::ColorRed,
        data_type: RecordDataType::U8,
    };

    pub const COLOR_GREEN_U8: Record = Record {
        name: RecordName::ColorGreen,
        data_type: RecordDataType::U8,
    };

    pub const COLOR_BLUE_U8: Record = Record {
        name: RecordName::ColorBlue,
        data_type: RecordDataType::U8,
    };

    pub const INTENSITY_U16: Record = Record {
        name: RecordName::Intensity,
        data_type: RecordDataType::U16,
    };

    pub const COLOR_RED_UNIT_F32: Record = Record {
        name: RecordName::ColorRed,
        data_type: RecordDataType::UNIT_F32,
    };

    pub const COLOR_GREEN_UNIT_F32: Record = Record {
        name: RecordName::ColorGreen,
        data_type: RecordDataType::UNIT_F32,
    };

    pub const COLOR_BLUE_UNIT_F32: Record = Record {
        name: RecordName::ColorBlue,
        data_type: RecordDataType::UNIT_F32,
    };

    pub const INTENSITY_UNIT_F32: Record = Record {
        name: RecordName::Intensity,
        data_type: RecordDataType::UNIT_F32,
    };
}
