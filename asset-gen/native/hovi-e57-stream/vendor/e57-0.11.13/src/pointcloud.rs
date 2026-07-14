use crate::error::Converter;
use crate::xml;
use crate::{
    CartesianBounds, ColorLimits, DateTime, IndexBounds, IntensityLimits, Record, RecordDataType,
    RecordName, Result, SphericalBounds, Transform,
};
use roxmltree::{Document, Node};

/// Descriptor with metadata for a single point cloud.
///
/// This struct does not contain any actual point data,
/// it just describes the properties and attributes of a point cloud.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct PointCloud {
    /// Globally unique identifier for the point cloud.
    /// Required by spec, but the reference C++ implementation does allow to omit this field, so we do too.
    pub guid: Option<String>,
    /// Physical file offset of the start of the associated binary section.
    pub file_offset: u64,
    /// Number of points in the point cloud.
    pub records: u64,
    /// List of point attributes that exist for this point cloud.
    pub prototype: Vec<Record>,
    /// Optional list of GUIDs that identify the data sets from which the points in this point cloud originated.
    /// Currently not used and always none. Included to avoid future breaking API changes.
    pub original_guids: Option<Vec<String>>,
    /// Optional user-defined name for the point cloud.
    pub name: Option<String>,
    /// Optional user-defined description of the point cloud.
    pub description: Option<String>,
    /// Optional axis-aligned Cartesian bounds for the point cloud in local coordinates.
    pub cartesian_bounds: Option<CartesianBounds>,
    /// Optional spherical bounds for the point cloud.
    pub spherical_bounds: Option<SphericalBounds>,
    /// Optional index bounds (row, column, return values) for the point cloud.
    pub index_bounds: Option<IndexBounds>,
    /// Optional intensity limits for the point cloud. Should represent the full range the sensor is able to capture.
    pub intensity_limits: Option<IntensityLimits>,
    /// Optional color limits for the point cloud. Should represent the full range the sensor is able to capture.
    pub color_limits: Option<ColorLimits>,
    /// Optional transformation to convert data from the local point cloud coordinates to the file-level coordinate system.
    pub transform: Option<Transform>,
    /// Optional start date and time when the point cloud was captured with a scanning device.
    pub acquisition_start: Option<DateTime>,
    /// Optional end date and time when the point cloud was captured with a scanning device.
    pub acquisition_end: Option<DateTime>,
    /// Optional name of the manufacturer for the sensor used to capture the point cloud.
    pub sensor_vendor: Option<String>,
    /// Optional model name of the sensor used for capturing.
    pub sensor_model: Option<String>,
    /// Optional serial number of the sensor used for capturing.
    pub sensor_serial: Option<String>,
    /// Optional version identifier for the sensor hardware used for capturing.
    pub sensor_hw_version: Option<String>,
    /// Optional version identifier for the sensor software used for capturing.
    pub sensor_sw_version: Option<String>,
    /// Optional version identifier for the sensor firmware used for capturing.
    pub sensor_fw_version: Option<String>,
    /// Optional ambient temperature in degrees Celsius, measured at the sensor at the time of capturing.
    pub temperature: Option<f64>,
    /// Optional percentage of relative humidity between 0 and 100, measured at the sensor at the time of capturing.
    pub humidity: Option<f64>,
    /// Optional atmospheric pressure in Pascals, measured at the sensor at the time of capturing.
    pub atmospheric_pressure: Option<f64>,
}

impl PointCloud {
    pub(crate) fn vec_from_document(document: &Document) -> Result<Vec<Self>> {
        let mut pointclouds = Vec::new();
        if let Some(data3d_node) = document.descendants().find(|n| n.has_tag_name("data3D")) {
            for n in data3d_node.children() {
                if n.has_tag_name("vectorChild") && n.attribute("type") == Some("Structure") {
                    let pointcloud = Self::from_node(&n)?;
                    pointclouds.push(pointcloud);
                }
            }
        }
        Ok(pointclouds)
    }

    pub(crate) fn from_node(node: &Node) -> Result<Self> {
        let guid = xml::opt_string(node, "guid")?;
        let name = xml::opt_string(node, "name")?;
        let description = xml::opt_string(node, "description")?;
        let sensor_model = xml::opt_string(node, "sensorModel")?;
        let sensor_vendor = xml::opt_string(node, "sensorVendor")?;
        let sensor_serial = xml::opt_string(node, "sensorSerialNumber")?;
        let sensor_hw_version = xml::opt_string(node, "sensorHardwareVersion")?;
        let sensor_sw_version = xml::opt_string(node, "sensorSoftwareVersion")?;
        let sensor_fw_version = xml::opt_string(node, "sensorFirmwareVersion")?;
        let temperature = xml::opt_f64(node, "temperature")?;
        let humidity = xml::opt_f64(node, "relativeHumidity")?;
        let atmospheric_pressure = xml::opt_f64(node, "atmosphericPressure")?;
        let acquisition_start = xml::opt_date_time(node, "acquisitionStart")?;
        let acquisition_end = xml::opt_date_time(node, "acquisitionEnd")?;
        let transform = xml::opt_transform(node, "pose")?;
        let cartesian_bounds = node.children().find(|n| n.has_tag_name("cartesianBounds"));
        let spherical_bounds = node.children().find(|n| n.has_tag_name("sphericalBounds"));
        let index_bounds = node.children().find(|n| n.has_tag_name("indexBounds"));
        let intensity_limits = node.children().find(|n| n.has_tag_name("intensityLimits"));
        let color_limits = node.children().find(|n| n.has_tag_name("colorLimits"));

        // Read optional vector of original GUIDs
        let original_guids = if let Some(original_guids_node) =
            node.children().find(|n| n.has_tag_name("originalGuids"))
        {
            let mut guids = Vec::new();
            for n in original_guids_node.children() {
                if !n.is_element()
                    || !n.has_tag_name("vectorChild")
                    || n.attribute("type") != Some("String")
                {
                    continue;
                }
                guids.push(n.text().unwrap_or("").to_owned())
            }
            Some(guids)
        } else {
            None
        };

        let points_tag = node
            .children()
            .find(|n| n.has_tag_name("points") && n.attribute("type") == Some("CompressedVector"))
            .invalid_err("Cannot find 'points' tag inside 'data3D' child")?;
        let file_offset = points_tag
            .attribute("fileOffset")
            .invalid_err("Cannot find 'fileOffset' attribute in 'points' tag")?
            .parse::<u64>()
            .invalid_err("Cannot parse 'fileOffset' attribute value as u64")?;
        let records = points_tag
            .attribute("recordCount")
            .invalid_err("Cannot find 'recordCount' attribute in 'points' tag")?
            .parse::<u64>()
            .invalid_err("Cannot parse 'recordCount' attribute value as u64")?;
        let prototype_tag = points_tag
            .children()
            .find(|n| n.has_tag_name("prototype") && n.attribute("type") == Some("Structure"))
            .invalid_err("Cannot find 'prototype' child in 'points' tag")?;

        // Parse point prototype records
        let mut prototype = Vec::new();
        for n in prototype_tag.children() {
            if !n.is_element() {
                continue;
            }
            let namespace_uri = n.tag_name().namespace();
            let ns = n.lookup_prefix(namespace_uri.unwrap_or_default());
            let tag = n.tag_name().name();
            let name = RecordName::from_namespace_and_tag_name(ns, namespace_uri, tag)?;
            let data_type = RecordDataType::from_node(&n)?;
            prototype.push(Record { name, data_type });
        }

        Ok(Self {
            guid,
            name,
            file_offset,
            original_guids,
            records,
            prototype,
            cartesian_bounds: if let Some(node) = cartesian_bounds {
                Some(CartesianBounds::from_node(&node)?)
            } else {
                None
            },
            spherical_bounds: if let Some(node) = spherical_bounds {
                Some(SphericalBounds::from_node(&node)?)
            } else {
                None
            },
            index_bounds: if let Some(node) = index_bounds {
                Some(IndexBounds::from_node(&node)?)
            } else {
                None
            },
            intensity_limits: if let Some(node) = intensity_limits {
                Some(IntensityLimits::from_node(&node)?)
            } else {
                None
            },
            color_limits: if let Some(node) = color_limits {
                Some(ColorLimits::from_node(&node)?)
            } else {
                None
            },
            transform,
            description,
            acquisition_start,
            acquisition_end,
            sensor_vendor,
            sensor_model,
            sensor_serial,
            sensor_hw_version,
            sensor_sw_version,
            sensor_fw_version,
            temperature,
            humidity,
            atmospheric_pressure,
        })
    }

    #[allow(clippy::cognitive_complexity)]
    pub(crate) fn xml_string(&self) -> Result<String> {
        let mut xml = String::new();
        xml += "<vectorChild type=\"Structure\">\n";
        if let Some(guid) = &self.guid {
            xml += &xml::gen_string("guid", &guid);
        }
        if let Some(original_guids) = &self.original_guids {
            xml += "<originalGuids type=\"Vector\" allowHeterogeneousChildren=\"0\">\n";
            for guid in original_guids {
                xml += &xml::gen_string("vectorChild", &guid);
            }
            xml += "</originalGuids>\n";
        }

        if let Some(bounds) = &self.cartesian_bounds {
            xml += &bounds.xml_string();
        }
        if let Some(bounds) = &self.spherical_bounds {
            xml += &bounds.xml_string();
        }
        if let Some(bounds) = &self.index_bounds {
            xml += &bounds.xml_string();
        }

        if let Some(limits) = &self.color_limits {
            // All members of the color limits struct are required,
            // so we only write the XML if we have all of them!
            if limits.red_min.is_some()
                && limits.red_max.is_some()
                && limits.green_min.is_some()
                && limits.green_max.is_some()
                && limits.blue_min.is_some()
                && limits.blue_max.is_some()
            {
                xml += &limits.xml_string();
            }
        }
        if let Some(limits) = &self.intensity_limits {
            // All members of the intensity limits struct are required,
            // so we only write the XML if we have all of them!
            if limits.intensity_min.is_some() && limits.intensity_max.is_some() {
                xml += &limits.xml_string();
            }
        }

        if let Some(name) = &self.name {
            xml += &xml::gen_string("name", name);
        }
        if let Some(desc) = &self.description {
            xml += &xml::gen_string("description", desc);
        }
        if let Some(sensor_vendor) = &self.sensor_vendor {
            xml += &xml::gen_string("sensorVendor", sensor_vendor);
        }
        if let Some(sensor_model) = &self.sensor_model {
            xml += &xml::gen_string("sensorModel", sensor_model);
        }
        if let Some(sensor_serial) = &self.sensor_serial {
            xml += &xml::gen_string("sensorSerialNumber", sensor_serial);
        }
        if let Some(sensor_sw_version) = &self.sensor_sw_version {
            xml += &xml::gen_string("sensorSoftwareVersion", sensor_sw_version);
        }
        if let Some(sensor_fw_version) = &self.sensor_fw_version {
            xml += &xml::gen_string("sensorFirmwareVersion", sensor_fw_version);
        }
        if let Some(sensor_hw_version) = &self.sensor_hw_version {
            xml += &xml::gen_string("sensorHardwareVersion", sensor_hw_version);
        }

        if let Some(transform) = &self.transform {
            xml += &transform.xml_string("pose");
        }
        if let Some(aq_start) = &self.acquisition_start {
            xml += &aq_start.xml_string("acquisitionStart");
        }
        if let Some(aq_end) = &self.acquisition_end {
            xml += &aq_end.xml_string("acquisitionEnd");
        }

        if let Some(temperature) = self.temperature {
            xml += &xml::gen_float("temperature", temperature);
        }
        if let Some(humidity) = self.humidity {
            xml += &xml::gen_float("relativeHumidity", humidity);
        }
        if let Some(pressure) = self.atmospheric_pressure {
            xml += &xml::gen_float("atmosphericPressure", pressure);
        }

        xml += &format!(
            "<points type=\"CompressedVector\" fileOffset=\"{}\" recordCount=\"{}\">\n",
            self.file_offset, self.records
        );
        xml += "<prototype type=\"Structure\">\n";
        for record in &self.prototype {
            xml += &record.xml_string();
        }
        xml += "</prototype>\n";
        xml += "</points>\n";

        xml += "</vectorChild>\n";
        Ok(xml)
    }

    fn contains(&self, names: &[RecordName]) -> bool {
        names
            .iter()
            .all(|searched| self.prototype.iter().any(|actual| &actual.name == searched))
    }

    /// Returns true if the point prototype contains X, Y and Z records for Cartesian coordinates.
    pub fn has_cartesian(&self) -> bool {
        self.contains(&[
            RecordName::CartesianX,
            RecordName::CartesianY,
            RecordName::CartesianZ,
        ])
    }

    /// Returns true if the point prototype contains range, azimuth and elevation records for Spherical coordinates.
    pub fn has_spherical(&self) -> bool {
        self.contains(&[
            RecordName::SphericalRange,
            RecordName::SphericalAzimuth,
            RecordName::SphericalElevation,
        ])
    }

    /// Returns true if the point prototype contains red, green and blue records for colors.
    pub fn has_color(&self) -> bool {
        self.contains(&[
            RecordName::ColorRed,
            RecordName::ColorGreen,
            RecordName::ColorBlue,
        ])
    }

    /// Returns true if the point prototype contains a intensity record.
    pub fn has_intensity(&self) -> bool {
        self.contains(&[RecordName::Intensity])
    }

    /// Returns true if the point prototype contains row and column index records.
    pub fn has_row_column(&self) -> bool {
        self.contains(&[RecordName::RowIndex, RecordName::ColumnIndex])
    }

    /// Returns true if the point prototype contains return count and return index records.
    pub fn has_return(&self) -> bool {
        self.contains(&[RecordName::ReturnCount, RecordName::ReturnIndex])
    }

    /// Returns true if the point prototype contains a time stamp record.
    pub fn has_timestamp(&self) -> bool {
        self.contains(&[RecordName::TimeStamp])
    }

    /// Tries to returns the Cartesian bounds of the point cloud, if possible.
    /// The bounds will be in local coordinates without the point clouds transformation pose applied.
    /// If existing, it will return the Cartesian bounds provided in the E57 file.
    /// If thats not possible it will try to calculate the bounds from any
    /// existing spherical bounds in the E57 file (see [`SphericalBounds::to_cartesian`]).
    /// If that is also not possible, it will return `None`.
    pub fn get_cartesian_bounds(&self) -> Option<CartesianBounds> {
        if let Some(bounds) = &self.cartesian_bounds {
            Some(bounds.clone())
        } else if let Some(bounds) = &self.spherical_bounds {
            bounds.to_cartesian()
        } else {
            None
        }
    }
}
