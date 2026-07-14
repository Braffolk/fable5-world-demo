use crate::error::Converter;
use crate::Result;
use roxmltree::Node;

/// Represents a specific date and time used in E57 files.
#[derive(Clone, Debug)]
pub struct DateTime {
    /// Number of seconds since GPS start epoch (00:00 UTC on January 6, 1980).
    pub gps_time: f64,
    /// True if the a global navigation satellite system device (such as GPS or GLONASS) was used to record the time.
    pub atomic_reference: bool,
}

impl DateTime {
    pub(crate) fn from_node(node: &Node) -> Result<Option<Self>> {
        let gps_time_text = node
            .children()
            .find(|n| n.has_tag_name("dateTimeValue") && n.attribute("type") == Some("Float"))
            .invalid_err("Unable to find XML tag 'dateTimeValue' with type 'Float'")?
            .text();
        let gps_time = if let Some(text) = gps_time_text {
            text.parse::<f64>()
                .invalid_err("Failed to parse inner text of XML tag 'dateTimeValue' as double")?
        } else {
            return Ok(None);
        };

        let atomic_reference_node = node.children().find(|n| {
            n.has_tag_name("isAtomicClockReferenced") && n.attribute("type") == Some("Integer")
        });
        let atomic_reference = if let Some(node) = atomic_reference_node {
            node.text().unwrap_or("0").trim() == "1"
        } else {
            return Ok(None);
        };

        Ok(Some(Self {
            gps_time,
            atomic_reference,
        }))
    }

    pub(crate) fn xml_string(&self, tag_name: &str) -> String {
        let mut xml = String::new();
        xml += &format!("<{tag_name} type=\"Structure\">\n");
        xml += &format!(
            "<dateTimeValue type=\"Float\">{}</dateTimeValue>\n",
            self.gps_time
        );
        xml += &format!(
            "<isAtomicClockReferenced type=\"Integer\">{}</isAtomicClockReferenced>\n",
            if self.atomic_reference { "1" } else { "0" }
        );
        xml += &format!("</{tag_name}>\n");
        xml
    }
}
