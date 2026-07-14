use crate::error::Converter;
use crate::Error;
use crate::RecordDataType;
use crate::RecordValue;
use crate::Result;
use roxmltree::Node;

fn extract_limit(bounds: &Node, tag_name: &str) -> Result<Option<RecordValue>> {
    if let Some(tag) = bounds.descendants().find(|n| n.has_tag_name(tag_name)) {
        let type_str = tag
            .attribute("type")
            .invalid_err(format!("Cannot find type attribute of limit '{tag_name}'"))?;
        let value_str = tag.text().unwrap_or("0");
        Ok(match type_str {
            "Integer" => Some(RecordValue::Integer(
                value_str
                    .parse::<i64>()
                    .invalid_err("Cannot parse integer limit value")?,
            )),
            "ScaledInteger" => Some(RecordValue::ScaledInteger(
                value_str
                    .parse::<i64>()
                    .invalid_err("Cannot parse scaled integer limit value")?,
            )),
            "Float" => {
                let single = tag.attribute("precision").unwrap_or("double") == "single";
                if single {
                    Some(RecordValue::Single(
                        value_str
                            .parse::<f32>()
                            .invalid_err("Cannot parse single limit value")?,
                    ))
                } else {
                    Some(RecordValue::Double(
                        value_str
                            .parse::<f64>()
                            .invalid_err("Cannot parse double limit value")?,
                    ))
                }
            }
            _ => Error::not_implemented(format!(
                "Found unsupported limit of type '{type_str}' for '{tag_name}'"
            ))?,
        })
    } else {
        Ok(None)
    }
}

/// Optional minimum and maximum values for intensity.
#[derive(Clone, Debug)]
pub struct IntensityLimits {
    pub intensity_min: Option<RecordValue>,
    pub intensity_max: Option<RecordValue>,
}

impl IntensityLimits {
    pub(crate) fn from_node(node: &Node) -> Result<Self> {
        let intensity_min = extract_limit(node, "intensityMinimum")?;
        let intensity_max = extract_limit(node, "intensityMaximum")?;
        Ok(Self {
            intensity_min,
            intensity_max,
        })
    }

    pub(crate) fn from_record_type(data_type: &RecordDataType) -> Self {
        let (intensity_min, intensity_max) = data_type.limits();
        Self {
            intensity_min,
            intensity_max,
        }
    }

    pub(crate) fn xml_string(&self) -> String {
        let mut xml = String::from("<intensityLimits type=\"Structure\">\n");
        if let Some(min) = &self.intensity_min {
            xml += &record_value_to_xml("intensityMinimum", min);
        }
        if let Some(max) = &self.intensity_max {
            xml += &record_value_to_xml("intensityMaximum", max);
        }
        xml += "</intensityLimits>\n";
        xml
    }
}

/// Optional minimum and maximum values for the colors red, green and blue.
#[derive(Clone, Debug)]
pub struct ColorLimits {
    pub red_min: Option<RecordValue>,
    pub red_max: Option<RecordValue>,
    pub green_min: Option<RecordValue>,
    pub green_max: Option<RecordValue>,
    pub blue_min: Option<RecordValue>,
    pub blue_max: Option<RecordValue>,
}

impl ColorLimits {
    pub(crate) fn from_node(node: &Node) -> Result<Self> {
        let red_min = extract_limit(node, "colorRedMinimum")?;
        let red_max = extract_limit(node, "colorRedMaximum")?;
        let green_min = extract_limit(node, "colorGreenMinimum")?;
        let green_max = extract_limit(node, "colorGreenMaximum")?;
        let blue_min = extract_limit(node, "colorBlueMinimum")?;
        let blue_max = extract_limit(node, "colorBlueMaximum")?;
        Ok(Self {
            red_min,
            red_max,
            green_min,
            green_max,
            blue_min,
            blue_max,
        })
    }

    pub(crate) fn from_record_types(
        red: &RecordDataType,
        green: &RecordDataType,
        blue: &RecordDataType,
    ) -> Self {
        let (red_min, red_max) = red.limits();
        let (green_min, green_max) = green.limits();
        let (blue_min, blue_max) = blue.limits();
        Self {
            red_min,
            red_max,
            green_min,
            green_max,
            blue_min,
            blue_max,
        }
    }

    pub(crate) fn xml_string(&self) -> String {
        let mut xml = String::from("<colorLimits type=\"Structure\">\n");
        if let Some(min) = &self.red_min {
            xml += &record_value_to_xml("colorRedMinimum", min);
        }
        if let Some(max) = &self.red_max {
            xml += &record_value_to_xml("colorRedMaximum", max);
        }
        if let Some(min) = &self.green_min {
            xml += &record_value_to_xml("colorGreenMinimum", min);
        }
        if let Some(max) = &self.green_max {
            xml += &record_value_to_xml("colorGreenMaximum", max);
        }
        if let Some(min) = &self.blue_min {
            xml += &record_value_to_xml("colorBlueMinimum", min);
        }
        if let Some(max) = &self.blue_max {
            xml += &record_value_to_xml("colorBlueMaximum", max);
        }
        xml += "</colorLimits>\n";
        xml
    }
}

/// Converts a record value to a XML limit tag with the correct type
fn record_value_to_xml(tag_name: &str, value: &RecordValue) -> String {
    match value {
        RecordValue::Integer(value) => {
            format!("<{tag_name} type=\"Integer\">{value}</{tag_name}>\n")
        }
        RecordValue::ScaledInteger(value) => {
            format!("<{tag_name} type=\"ScaledInteger\">{value}</{tag_name}>\n")
        }
        RecordValue::Single(value) => {
            format!("<{tag_name} type=\"Float\" precision=\"single\">{value}</{tag_name}>\n")
        }
        RecordValue::Double(value) => format!("<{tag_name} type=\"Float\">{value}</{tag_name}>\n"),
    }
}
