use crate::bs_write::ByteStreamWriteBuffer;
use crate::cv_section::CompressedVectorSectionHeader;
use crate::error::Converter;
use crate::packet::DataPacketHeader;
use crate::paged_writer::PagedWriter;
use crate::CartesianBounds;
use crate::ColorLimits;
use crate::DateTime;
use crate::Error;
use crate::IndexBounds;
use crate::IntensityLimits;
use crate::PointCloud;
use crate::RawValues;
use crate::Record;
use crate::RecordDataType;
use crate::RecordName;
use crate::RecordValue;
use crate::Result;
use crate::SphericalBounds;
use crate::Transform;
use std::collections::VecDeque;
use std::io::{Read, Seek, Write};

/// Creates a new point cloud by taking points and writing them into an E57 file.
///
/// Bounds for Cartesian and spherical coordinates are extracted automatically.
/// The same is also true for any index bounds.
pub struct PointCloudWriter<'a, T: Read + Write + Seek> {
    writer: &'a mut PagedWriter<T>,
    pointclouds: &'a mut Vec<PointCloud>,
    guid: String,
    section_offset: u64,
    section_header: CompressedVectorSectionHeader,
    original_guids: Option<Vec<String>>,
    prototype: Vec<Record>,
    cartesian_invalid_index: Option<usize>,
    spherical_invalid_index: Option<usize>,
    point_count: u64,
    buffer: VecDeque<RawValues>,
    max_points_per_packet: usize,
    byte_streams: Vec<ByteStreamWriteBuffer>,
    cartesian_bounds: Option<CartesianBounds>,
    spherical_bounds: Option<SphericalBounds>,
    index_bounds: Option<IndexBounds>,
    color_limits: Option<ColorLimits>,
    intensity_limits: Option<IntensityLimits>,
    name: Option<String>,
    description: Option<String>,
    transform: Option<Transform>,
    acquisition_start: Option<DateTime>,
    acquisition_end: Option<DateTime>,
    sensor_vendor: Option<String>,
    sensor_model: Option<String>,
    sensor_serial: Option<String>,
    sensor_hw_version: Option<String>,
    sensor_sw_version: Option<String>,
    sensor_fw_version: Option<String>,
    temperature: Option<f64>,
    humidity: Option<f64>,
    atmospheric_pressure: Option<f64>,
}

impl<'a, T: Read + Write + Seek> PointCloudWriter<'a, T> {
    pub(crate) fn new(
        writer: &'a mut PagedWriter<T>,
        pointclouds: &'a mut Vec<PointCloud>,
        guid: &str,
        prototype: Vec<Record>,
    ) -> Result<Self> {
        // Make sure the prototype is not invalid or incomplete
        Self::validate_prototype(&prototype)?;

        // Calculate max number of points per packet
        let max_points_per_packet = get_max_packet_points(&prototype);

        // Prepare byte stream buffers
        let byte_streams = vec![ByteStreamWriteBuffer::new(); prototype.len()];

        // Write preliminary section header with incomplete length and wrong offsets
        let mut section_header = CompressedVectorSectionHeader::default();
        let section_offset = writer.physical_position()?;
        section_header.section_length = CompressedVectorSectionHeader::SIZE;
        section_header.write(writer)?;

        // Now we know the data offset and can set it for later
        section_header.data_offset = writer.physical_position()?;

        // Prepare bounds
        let has_cartesian = prototype.iter().any(|p| p.name == RecordName::CartesianX);
        let cartesian_bounds = if has_cartesian {
            Some(CartesianBounds::default())
        } else {
            None
        };
        let has_spherical = prototype
            .iter()
            .any(|p| p.name == RecordName::SphericalAzimuth);
        let spherical_bounds = if has_spherical {
            Some(SphericalBounds::default())
        } else {
            None
        };
        let has_index = prototype.iter().any(|p| {
            p.name == RecordName::ReturnIndex
                || p.name == RecordName::ColumnIndex
                || p.name == RecordName::RowIndex
        });
        let index_bounds = if has_index {
            Some(IndexBounds::default())
        } else {
            None
        };

        // Prepare limits
        let has_color = prototype.iter().any(|p| p.name == RecordName::ColorRed);
        let color_limits = if has_color {
            let red_record = prototype
                .iter()
                .find(|p| p.name == RecordName::ColorRed)
                .internal_err("Unable to find red record")?;
            let green_record = prototype
                .iter()
                .find(|p| p.name == RecordName::ColorGreen)
                .internal_err("Unable to find green record")?;
            let blue_record = prototype
                .iter()
                .find(|p| p.name == RecordName::ColorBlue)
                .internal_err("Unable to find blue record")?;
            Some(ColorLimits::from_record_types(
                &red_record.data_type,
                &green_record.data_type,
                &blue_record.data_type,
            ))
        } else {
            None
        };
        let intensity = prototype.iter().find(|p| p.name == RecordName::Intensity);
        let intensity_limits = intensity.map(|i| IntensityLimits::from_record_type(&i.data_type));

        // Get prototype index positions of invalid states, if available.
        // They are needed later to skip bounds extraction for invalid points.
        let cartesian_invalid_index = prototype
            .iter()
            .enumerate()
            .find(|p| p.1.name == RecordName::CartesianInvalidState)
            .map(|p| p.0);
        let spherical_invalid_index = prototype
            .iter()
            .enumerate()
            .find(|p| p.1.name == RecordName::SphericalInvalidState)
            .map(|p| p.0);

        Ok(PointCloudWriter {
            writer,
            pointclouds,
            guid: guid.to_owned(),
            section_offset,
            section_header,
            original_guids: None,
            prototype,
            cartesian_invalid_index,
            spherical_invalid_index,
            point_count: 0,
            buffer: VecDeque::new(),
            byte_streams,
            max_points_per_packet,
            cartesian_bounds,
            spherical_bounds,
            index_bounds,
            color_limits,
            intensity_limits,
            name: None,
            description: None,
            transform: None,
            acquisition_start: None,
            acquisition_end: None,
            sensor_vendor: None,
            sensor_model: None,
            sensor_serial: None,
            sensor_hw_version: None,
            sensor_sw_version: None,
            sensor_fw_version: None,
            temperature: None,
            humidity: None,
            atmospheric_pressure: None,
        })
    }

    /// Set optional user-defined name for the point cloud (empty by default).
    pub fn set_name(&mut self, value: Option<String>) {
        self.name = value;
    }

    /// Set optional user-defined description for the point cloud (empty by default).
    pub fn set_description(&mut self, value: Option<String>) {
        self.description = value;
    }

    /// Set optional original GUIDs to indicate the source point clouds used to create this one.
    /// This is useful to keep track which different point clouds were combined.
    pub fn set_original_guids(&mut self, value: Option<Vec<String>>) {
        self.original_guids = value;
    }

    /// Set optional transformation to convert data from the local
    /// point cloud coordinates to the file-level coordinate system.
    /// By default this is empty, meaning the point cloud has no transformation.
    pub fn set_transform(&mut self, value: Option<Transform>) {
        self.transform = value;
    }

    /// Set optional start date and time when the point cloud was
    /// captured with a scanning device (empty by default).
    pub fn set_acquisition_start(&mut self, value: Option<DateTime>) {
        self.acquisition_start = value;
    }

    /// Set optional end date and time when the point cloud was
    /// captured with a scanning device (empty by default).
    pub fn set_acquisition_end(&mut self, value: Option<DateTime>) {
        self.acquisition_end = value;
    }

    /// Set optional name of the manufacturer for the sensor used
    /// to capture the point cloud (empty by default).
    pub fn set_sensor_vendor(&mut self, value: Option<String>) {
        self.sensor_vendor = value;
    }

    /// Set optional model name of the sensor used for capturing (empty by default).
    pub fn set_sensor_model(&mut self, value: Option<String>) {
        self.sensor_model = value;
    }

    /// Set optional serial number of the sensor used for capturing (empty by default).
    pub fn set_sensor_serial(&mut self, value: Option<String>) {
        self.sensor_serial = value;
    }

    /// Set optional version identifier for the sensor software
    /// used for capturing (empty by default).
    pub fn set_sensor_sw_version(&mut self, value: Option<String>) {
        self.sensor_sw_version = value;
    }

    /// Set optional version identifier for the sensor hardware
    /// used for capturing (empty by default).
    pub fn set_sensor_hw_version(&mut self, value: Option<String>) {
        self.sensor_hw_version = value;
    }

    /// Set optional version identifier for the sensor firmware
    /// used for capturing (empty by default).
    pub fn set_sensor_fw_version(&mut self, value: Option<String>) {
        self.sensor_fw_version = value;
    }

    /// Set optional ambient temperature in degrees Celsius,
    /// measured at the sensor at the time of capturing (empty by default).
    pub fn set_temperature(&mut self, value: Option<f64>) {
        self.temperature = value;
    }

    /// Set optional percentage of relative humidity between 0 and 100,
    /// measured at the sensor at the time of capturing (empty by default).
    pub fn set_humidity(&mut self, value: Option<f64>) {
        self.humidity = value;
    }

    /// Set optional atmospheric pressure in Pascals,
    /// measured at the sensor at the time of capturing (empty by default).
    pub fn set_atmospheric_pressure(&mut self, value: Option<f64>) {
        self.atmospheric_pressure = value;
    }

    /// Sets the intensity limits of the point cloud.
    /// Only required if the point cloud contains intensity data.
    /// This must represent the full range of the sensor that captured the point cloud data.
    /// The default value is the range extracted from the prototypes intensity data type.
    /// Since the data type range might be bigger than the actual sensor range,
    /// it is strongly recommended to set this limit manually.
    pub fn set_intensity_limits(&mut self, value: Option<IntensityLimits>) {
        self.intensity_limits = value;
    }

    /// Sets the color limits of the point cloud.
    /// Only required if the point cloud contains color data.
    /// This must represent the full range of all possible color values.
    /// The default value is the range extracted from the prototypes color data types.
    /// Since the data type range might be different from the actual limits,
    /// it is strongly recommended to set this limit manually.
    pub fn set_color_limits(&mut self, value: Option<ColorLimits>) {
        self.color_limits = value;
    }

    /// Sets and overrides the index bounds of the point cloud manually.
    /// This should not be used in normales use cases, since the bounds
    /// are set and updated automatically when you add a point that contains
    /// a row, column or return value.
    /// If you need to set the bounds manually, this needs to be done **after**
    /// adding the last point and before calling `finalize()`!
    pub fn set_index_bounds(&mut self, value: Option<IndexBounds>) {
        self.index_bounds = value;
    }

    /// Sets and overrides the Cartesian bounds of the point cloud manually.
    /// This should not be used in normales use cases, since the bounds
    /// are set and updated automatically when you add a point that contains
    /// a Cartesian coordinate. Note that the Cartesian bounds are **not** updated
    /// when you add a point that only contains spherical coordinates.
    /// If you want additional Cartesian bounds in your E57 file,
    /// you need to extract them manually and set them using this method.
    /// If you set the bounds manually, this needs to be done **after**
    /// adding the last point and before calling `finalize()`!
    pub fn set_cartesian_bounds(&mut self, value: Option<CartesianBounds>) {
        self.cartesian_bounds = value;
    }

    /// Sets and overrides the spherical bounds of the point cloud manually.
    /// This should not be used in normales use cases, since the bounds
    /// are set and updated automatically when you add a point that contains
    /// a spherical coordinate.
    /// If you need to set the bounds manually, this needs to be done **after**
    /// adding the last point and before calling `finalize()`!
    pub fn set_spherical_bounds(&mut self, value: Option<SphericalBounds>) {
        self.spherical_bounds = value;
    }

    fn validate_prototype(prototype: &[Record]) -> Result<()> {
        // Helpers to check and look up records
        let contains = |n: RecordName| prototype.iter().any(|p| p.name == n);
        let get = |n: RecordName| prototype.iter().find(|p| p.name == n);

        // Cartesian or spherical?
        validate_cartesian(prototype)?;
        validate_spherical(prototype)?;
        if !contains(RecordName::CartesianX) && !contains(RecordName::SphericalAzimuth) {
            Error::invalid("You have to include Cartesian or spherical coordinates")?
        }

        validate_color(prototype)?;
        validate_return(prototype)?;

        // Row & column check
        if let Some(record) = get(RecordName::RowIndex) {
            match record.data_type {
                RecordDataType::Integer { .. } => {}
                _ => Error::invalid("RowIndex must have an integer type")?,
            }
        }
        if let Some(record) = get(RecordName::ColumnIndex) {
            match record.data_type {
                RecordDataType::Integer { .. } => {}
                _ => Error::invalid("ColumnIndex must have an integer type")?,
            }
        }

        // Intensity check
        if let Some(record) = get(RecordName::IsIntensityInvalid) {
            if !contains(RecordName::Intensity) {
                Error::invalid("IsIntensityInvalid requires Intensity")?
            }
            match record.data_type {
                RecordDataType::Integer { min: 0, max: 1 } => {}
                _ => Error::invalid("IsIntensityInvalid needs to be an integer between 0 and 1")?,
            }
        }

        // Time stamp check
        if let Some(record) = get(RecordName::IsTimeStampInvalid) {
            if !contains(RecordName::TimeStamp) {
                Error::invalid("IsTimeStampInvalid requires TimeStamp")?
            }
            match record.data_type {
                RecordDataType::Integer { min: 0, max: 1 } => {}
                _ => Error::invalid("IsTimeStampInvalid needs to be an integer between 0 and 1")?,
            }
        }

        Ok(())
    }

    fn write_buffer_to_disk(&mut self, last_flush: bool) -> Result<()> {
        // Add points from buffer into byte streams
        let packet_points = self.max_points_per_packet.min(self.buffer.len());
        let proto_len = self.prototype.len();
        for _ in 0..packet_points {
            let p = self
                .buffer
                .pop_front()
                .internal_err("Failed to get next point for writing")?;
            for (i, prototype) in self.prototype.iter().enumerate() {
                let raw_value = p
                    .get(i)
                    .invalid_err("Prototype is bigger than number of provided values")?;
                prototype
                    .data_type
                    .write(raw_value, &mut self.byte_streams[i])?;
            }
        }

        // Check and prepare buffer sizes
        let mut sum_bs_sizes = 0;
        let mut bs_sizes = Vec::with_capacity(proto_len);
        for bs in &self.byte_streams {
            let bs_size = if last_flush {
                bs.all_bytes()
            } else {
                bs.full_bytes()
            };
            sum_bs_sizes += bs_size;
            bs_sizes.push(bs_size as u16);
        }

        // Only write packet if there is actual data!
        if sum_bs_sizes > 0 {
            // Calculate packet length for header, must be aligned to four bytes.
            // If the length exceeds 2^16 this library has somewhere a logic bug!
            let mut packet_length = DataPacketHeader::SIZE + proto_len * 2 + sum_bs_sizes;
            if !packet_length.is_multiple_of(4) {
                let missing = 4 - (packet_length % 4);
                packet_length += missing;
            }
            if packet_length > u16::MAX as usize {
                Error::internal("Invalid data packet length detected")?
            }

            // Add data packet length to section length for later
            self.section_header.section_length += packet_length as u64;

            // Write data packet header
            DataPacketHeader {
                comp_restart_flag: false,
                packet_length: packet_length as u64,
                bytestream_count: proto_len as u16,
            }
            .write(&mut self.writer)?;

            // Write bytestream sizes as u16 values
            for size in bs_sizes {
                let bytes = size.to_le_bytes();
                self.writer
                    .write_all(&bytes)
                    .write_err("Cannot write data packet buffer size")?;
            }

            // Write actual bytestream buffers with data
            for bs in &mut self.byte_streams {
                let data = if last_flush {
                    bs.get_all_bytes()
                } else {
                    bs.get_full_bytes()
                };
                self.writer
                    .write_all(&data)
                    .write_err("Cannot write bytestream buffer into data packet")?;
            }
        }

        self.writer
            .align()
            .write_err("Failed to align writer on next 4-byte offset after writing data packet")?;

        Ok(())
    }

    /// Adds a new point to the point cloud.
    #[allow(clippy::cognitive_complexity)]
    pub fn add_point(&mut self, values: RawValues) -> Result<()> {
        if values.len() != self.prototype.len() {
            Error::invalid("Number of values does not match prototype length")?
        }

        // Go over all values to validate and extract min/max values
        for (i, p) in self.prototype.iter().enumerate() {
            let value = &values[i];

            // Ensure that each value fits the corresponding prototype entry
            if !match p.data_type {
                RecordDataType::Single { .. } => matches!(value, RecordValue::Single(..)),
                RecordDataType::Double { .. } => matches!(value, RecordValue::Double(..)),
                RecordDataType::ScaledInteger { .. } => {
                    matches!(value, RecordValue::ScaledInteger(..))
                }
                RecordDataType::Integer { .. } => matches!(value, RecordValue::Integer(..)),
            } {
                Error::invalid(format!(
                    "Type mismatch at index {i}: value type does not match prototype"
                ))?
            }

            // Update cartesian bounds
            if p.name == RecordName::CartesianX
                || p.name == RecordName::CartesianY
                || p.name == RecordName::CartesianZ
            {
                let value = values[i].to_f64(&p.data_type)?;

                // Check if the Cartesian coordinate is valid
                let valid = if let Some(index) = self.cartesian_invalid_index {
                    let value = values[index].to_i64(&self.prototype[index].data_type)?;
                    value == 0
                } else {
                    true
                };

                if valid {
                    let bounds = self
                        .cartesian_bounds
                        .as_mut()
                        .internal_err("Cannot find Cartesian bounds")?;
                    if p.name == RecordName::CartesianX {
                        update_min(value, &mut bounds.x_min);
                        update_max(value, &mut bounds.x_max);
                    }
                    if p.name == RecordName::CartesianY {
                        update_min(value, &mut bounds.y_min);
                        update_max(value, &mut bounds.y_max);
                    }
                    if p.name == RecordName::CartesianZ {
                        update_min(value, &mut bounds.z_min);
                        update_max(value, &mut bounds.z_max);
                    }
                }
            }

            // Update spherical bounds
            if p.name == RecordName::SphericalAzimuth
                || p.name == RecordName::SphericalElevation
                || p.name == RecordName::SphericalRange
            {
                let value = values[i].to_f64(&p.data_type)?;

                // Check if the spherical coordinate is valid
                let valid = if let Some(index) = self.spherical_invalid_index {
                    let value = values[index].to_i64(&self.prototype[index].data_type)?;
                    value == 0
                } else {
                    true
                };

                if valid {
                    let bounds = self
                        .spherical_bounds
                        .as_mut()
                        .internal_err("Cannot find spherical bounds")?;
                    if p.name == RecordName::SphericalAzimuth {
                        update_min(value, &mut bounds.azimuth_start);
                        update_max(value, &mut bounds.azimuth_end);
                    }
                    if p.name == RecordName::SphericalElevation {
                        update_min(value, &mut bounds.elevation_min);
                        update_max(value, &mut bounds.elevation_max);
                    }
                    // Range must be updated only for fully valid spherical coordinates
                    if p.name == RecordName::SphericalRange {
                        update_min(value, &mut bounds.range_min);
                        update_max(value, &mut bounds.range_max);
                    }
                }
            }

            // Update index bounds
            if p.name == RecordName::RowIndex
                || p.name == RecordName::ColumnIndex
                || p.name == RecordName::ReturnIndex
            {
                let value = values[i].to_i64(&p.data_type)?;
                let bounds = self
                    .index_bounds
                    .as_mut()
                    .internal_err("Cannot find index bounds")?;
                if p.name == RecordName::RowIndex {
                    update_min(value, &mut bounds.row_min);
                    update_max(value, &mut bounds.row_max);
                }
                if p.name == RecordName::ColumnIndex {
                    update_min(value, &mut bounds.column_min);
                    update_max(value, &mut bounds.column_max);
                }
                if p.name == RecordName::ReturnIndex {
                    update_min(value, &mut bounds.return_min);
                    update_max(value, &mut bounds.return_max);
                }
            }
        }

        // Add new point to output buffer
        self.buffer.push_back(values);
        self.point_count += 1;

        // Empty buffer and write points when its full
        if self.buffer.len() >= self.max_points_per_packet {
            self.write_buffer_to_disk(false)?;
        }

        Ok(())
    }

    /// Called after all points have been added to finalize the creation of the new point cloud.
    pub fn finalize(&mut self) -> Result<()> {
        // Flush remaining points from buffer into byte streams and write them
        while !self.buffer.is_empty() {
            self.write_buffer_to_disk(false)?;
        }

        // Flush last partial bytes from byte streams
        self.write_buffer_to_disk(true)?;

        // We need to write the section header again with the final length
        // which was previously unknown and is now available.
        let end_offset = self
            .writer
            .physical_position()
            .write_err("Failed to get section end offset")?;
        self.writer
            .physical_seek(self.section_offset)
            .write_err("Failed to seek to section start for final update")?;
        self.section_header.write(&mut self.writer)?;
        self.writer
            .physical_seek(end_offset)
            .write_err("Failed to seek behind finalized section")?;

        // prepare point cloud metadata
        let pc = PointCloud {
            guid: Some(self.guid.clone()),
            records: self.point_count,
            file_offset: self.section_offset,
            original_guids: self.original_guids.take(),
            prototype: self.prototype.clone(),
            cartesian_bounds: self.cartesian_bounds.take(),
            spherical_bounds: self.spherical_bounds.take(),
            index_bounds: self.index_bounds.take(),
            color_limits: self.color_limits.take(),
            intensity_limits: self.intensity_limits.take(),
            name: self.name.take(),
            description: self.description.take(),
            transform: self.transform.take(),
            acquisition_start: self.acquisition_start.take(),
            acquisition_end: self.acquisition_end.take(),
            sensor_vendor: self.sensor_vendor.take(),
            sensor_model: self.sensor_model.take(),
            sensor_serial: self.sensor_serial.take(),
            sensor_hw_version: self.sensor_hw_version.take(),
            sensor_sw_version: self.sensor_sw_version.take(),
            sensor_fw_version: self.sensor_fw_version.take(),
            temperature: self.temperature.take(),
            humidity: self.humidity.take(),
            atmospheric_pressure: self.atmospheric_pressure.take(),
        };

        // Add metadata for XML generation later, when the file is completed.
        self.pointclouds.push(pc);

        Ok(())
    }
}

fn update_min<T: PartialOrd>(value: T, min: &mut Option<T>) {
    if let Some(current) = min {
        if *current > value {
            *min = Some(value)
        }
    } else {
        *min = Some(value)
    }
}

fn update_max<T: PartialOrd>(value: T, min: &mut Option<T>) {
    if let Some(current) = min {
        if *current < value {
            *min = Some(value)
        }
    } else {
        *min = Some(value)
    }
}

fn contains(prototype: &[Record], name: RecordName) -> bool {
    prototype.iter().any(|p| p.name == name)
}

fn get(prototype: &[Record], name: RecordName) -> Option<&Record> {
    prototype.iter().find(|p| p.name == name)
}

/// Validate Cartesian coordinates in prototype
fn validate_cartesian(prototype: &[Record]) -> Result<()> {
    let mut cartesian = 0;
    if contains(prototype, RecordName::CartesianX) {
        cartesian += 1;
    }
    if contains(prototype, RecordName::CartesianY) {
        cartesian += 1;
    }
    if contains(prototype, RecordName::CartesianZ) {
        cartesian += 1;
    }
    if cartesian != 0 && cartesian != 3 {
        Error::invalid("You have to include all three Cartesian coordinates for X, Y and Z")?
    }
    if let Some(record) = get(prototype, RecordName::CartesianInvalidState) {
        if !contains(prototype, RecordName::CartesianX) {
            Error::invalid("CartesianInvalidState requires Cartesian coordinates")?
        }
        match record.data_type {
            RecordDataType::Integer { min: 0, max: 2 } => {}
            _ => Error::invalid("CartesianInvalidState needs to be an integer between 0 and 2")?,
        }
    }
    Ok(())
}

/// Validate spherical coordinates in prototype
fn validate_spherical(prototype: &[Record]) -> Result<()> {
    let mut spherical = 0;
    if contains(prototype, RecordName::SphericalAzimuth) {
        spherical += 1;
    }
    if contains(prototype, RecordName::SphericalElevation) {
        spherical += 1;
    }
    if contains(prototype, RecordName::SphericalRange) {
        spherical += 1;
    }
    if spherical != 0 && spherical != 3 {
        Error::invalid(
            "You have to include all three spherical coordinates for azimuth, elevation and range",
        )?
    }
    if let Some(record) = get(prototype, RecordName::SphericalInvalidState) {
        if !contains(prototype, RecordName::SphericalAzimuth) {
            Error::invalid("SphericalInvalidState requires spherical coordinates")?
        }
        match record.data_type {
            RecordDataType::Integer { min: 0, max: 2 } => {}
            _ => Error::invalid("SphericalInvalidState needs to be an integer between 0 and 2")?,
        }
    }
    if let Some(record) = get(prototype, RecordName::SphericalAzimuth) {
        if let RecordDataType::Integer { .. } = record.data_type {
            Error::invalid("SphericalAzimuth cannot have an integer type")?
        }
    }
    if let Some(record) = get(prototype, RecordName::SphericalElevation) {
        if let RecordDataType::Integer { .. } = record.data_type {
            Error::invalid("SphericalElevation cannot have an integer type")?
        }
    }
    Ok(())
}

/// Validate color in prototype
fn validate_color(prototype: &[Record]) -> Result<()> {
    let mut color = 0;
    if contains(prototype, RecordName::ColorRed) {
        color += 1;
    }
    if contains(prototype, RecordName::ColorGreen) {
        color += 1;
    }
    if contains(prototype, RecordName::ColorBlue) {
        color += 1;
    }
    if color != 0 && color != 3 {
        Error::invalid("You have to include all three color values for red, green and blue")?
    }
    if let Some(record) = get(prototype, RecordName::IsColorInvalid) {
        if !contains(prototype, RecordName::ColorRed) {
            Error::invalid("IsColorInvalid requires colors")?
        }
        match record.data_type {
            RecordDataType::Integer { min: 0, max: 1 } => {}
            _ => Error::invalid("IsColorInvalid needs to be an integer between 0 and 1")?,
        }
    }
    Ok(())
}

/// Validate return in prototype
fn validate_return(prototype: &[Record]) -> Result<()> {
    let mut ret = 0;
    if let Some(record) = get(prototype, RecordName::ReturnCount) {
        ret += 1;
        match record.data_type {
            RecordDataType::Integer { .. } => {}
            _ => Error::invalid("ReturnCount must have an integer type")?,
        }
    }
    if let Some(record) = get(prototype, RecordName::ReturnIndex) {
        ret += 1;
        match record.data_type {
            RecordDataType::Integer { .. } => {}
            _ => Error::invalid("ReturnIndex must have an integer type")?,
        }
    }
    if ret != 0 && ret != 2 {
        Error::invalid("You have to include both, ReturnCount and ReturnIndex")?
    }
    Ok(())
}

/// Calculate maximum number of points per packet.
/// Each data packet can contain up to 2^16 bytes, but we need some reserved
/// space for header data. We also need to consider some "incomplete" bytes
/// from record value sizes that are not a multiple of 8 bits.
fn get_max_packet_points(prototype: &[Record]) -> usize {
    const SAFETY_MARGIN: usize = 500;
    let point_size_bits: usize = prototype.iter().map(|p| p.data_type.bit_size()).sum();
    let bs_size_headers = prototype.len() * 2; // u16 for each byte stream header
    let headers_size = DataPacketHeader::SIZE + bs_size_headers;
    let max_incomplete_bytes = prototype.len();
    let u16_max = u16::MAX as usize;
    ((u16_max - headers_size - max_incomplete_bytes - SAFETY_MARGIN) * 8) / point_size_bits
}
