use crate::paged_writer::PagedWriter;
use crate::Blob;
use crate::CylindricalImage;
use crate::CylindricalImageProperties;
use crate::DateTime;
use crate::Error;
use crate::Image;
use crate::ImageBlob;
use crate::ImageFormat;
use crate::PinholeImage;
use crate::PinholeImageProperties;
use crate::Projection;
use crate::Result;
use crate::SphericalImage;
use crate::SphericalImageProperties;
use crate::Transform;
use crate::VisualReferenceImage;
use crate::VisualReferenceImageProperties;
use std::io::{Read, Seek, Write};

/// Defines a new image and writes it into an E57 file.
pub struct ImageWriter<'a, T: Read + Write + Seek> {
    writer: &'a mut PagedWriter<T>,
    images: &'a mut Vec<Image>,
    image: Image,
}

impl<'a, T: Read + Write + Seek> ImageWriter<'a, T> {
    pub(crate) fn new(
        writer: &'a mut PagedWriter<T>,
        images: &'a mut Vec<Image>,
        guid: &str,
    ) -> Result<Self> {
        Ok(Self {
            writer,
            images,
            image: Image {
                guid: Some(guid.to_owned()),
                visual_reference: None,
                projection: None,
                transform: None,
                pointcloud_guid: None,
                name: None,
                description: None,
                acquisition: None,
                sensor_vendor: None,
                sensor_model: None,
                sensor_serial: None,
            },
        })
    }

    /// Set optional user-defined name for the image.
    /// Not set by default.
    pub fn set_name(&mut self, value: &str) {
        self.image.name = Some(value.to_owned());
    }

    /// Set optional user-defined description for the image.
    /// Not set by default.
    pub fn set_description(&mut self, value: &str) {
        self.image.description = Some(value.to_owned());
    }

    /// Set optional GUID of the point cloud that is connected to this image.
    /// Not set by default.
    pub fn set_pointcloud_guid(&mut self, value: &str) {
        self.image.pointcloud_guid = Some(value.to_owned());
    }

    /// Set optional transformation to convert data from the local
    /// image coordinates to the file-level coordinate system.
    /// By default this is not set, meaning the image has no transformation.
    pub fn set_transform(&mut self, value: Transform) {
        self.image.transform = Some(value);
    }

    /// Set optional start date and time when the images was captured.
    /// Not set by default.
    pub fn set_acquisition(&mut self, value: DateTime) {
        self.image.acquisition = Some(value);
    }

    /// Set optional name of the manufacturer for the sensor used to capture the image.
    /// Not set by default.
    pub fn set_sensor_vendor(&mut self, value: &str) {
        self.image.sensor_vendor = Some(value.to_owned());
    }

    /// Set optional model name of the sensor used for capturing the image.
    /// Not set by default.
    pub fn set_sensor_model(&mut self, value: &str) {
        self.image.sensor_model = Some(value.to_owned());
    }

    /// Set optional serial number of the sensor used for capturing the image.
    /// Not set by default.
    pub fn set_sensor_serial(&mut self, value: &str) {
        self.image.sensor_serial = Some(value.to_owned());
    }

    /// Adds an optional visual reference image, also known as preview image.
    /// See also `VisualReferenceImageProperties` struct for more details.
    /// The optional PNG mask image can be used to indicate valid/invalid
    /// pixels in the image, for example if the image is not rectangular.
    /// The mask must have the same size as the actual image.
    /// Non-zero-valued pixels mark valid pixel locations and
    /// zero-valued pixels mark invalid pixels.
    pub fn add_visual_reference(
        &mut self,
        format: ImageFormat,
        image: &mut dyn Read,
        properties: VisualReferenceImageProperties,
        mask: Option<&mut dyn Read>,
    ) -> Result<()> {
        let data = Blob::write(self.writer, image)?;
        let blob = ImageBlob { data, format };
        let mask = if let Some(mask_data) = mask {
            Some(Blob::write(self.writer, mask_data)?)
        } else {
            None
        };
        self.image.visual_reference = Some(VisualReferenceImage {
            properties,
            mask,
            blob,
        });
        Ok(())
    }

    /// Adds pinhole image data.
    /// Width and height must match the actual binary PNG or JPEG image.
    /// See also `PinholeImageProperties` struct for more details.
    /// The optional PNG mask image can be used to indicate valid/invalid
    /// pixels in the image, for example if the image is not rectangular.
    /// The mask must have the same size as the actual image.
    /// Non-zero-valued pixels mark valid pixel locations and
    /// zero-valued pixels mark invalid pixels.
    pub fn add_pinhole(
        &mut self,
        format: ImageFormat,
        image: &mut dyn Read,
        properties: PinholeImageProperties,
        mask: Option<&mut dyn Read>,
    ) -> Result<()> {
        if self.image.projection.is_some() {
            Error::invalid("A projected image is already set")?
        }
        let data = Blob::write(self.writer, image)?;
        let blob = ImageBlob { data, format };
        let mask = if let Some(mask_data) = mask {
            Some(Blob::write(self.writer, mask_data)?)
        } else {
            None
        };
        let rep = PinholeImage {
            blob,
            mask,
            properties,
        };
        self.image.projection = Some(Projection::Pinhole(rep));
        Ok(())
    }

    /// Adds spherical image data.
    /// See also `SphericalImageProperties` struct for more details.
    /// The optional PNG mask image can be used to indicate valid/invalid
    /// pixels in the image, for example if the image is not rectangular.
    /// The mask must have the same size as the actual image.
    /// Non-zero-valued pixels mark valid pixel locations and
    /// zero-valued pixels mark invalid pixels.
    pub fn add_spherical(
        &mut self,
        format: ImageFormat,
        image: &mut dyn Read,
        properties: SphericalImageProperties,
        mask: Option<&mut dyn Read>,
    ) -> Result<()> {
        if self.image.projection.is_some() {
            Error::invalid("A projected image is already set")?
        }
        let data = Blob::write(self.writer, image)?;
        let blob = ImageBlob { data, format };
        let mask = if let Some(mask_data) = mask {
            Some(Blob::write(self.writer, mask_data)?)
        } else {
            None
        };
        let rep = SphericalImage {
            blob,
            mask,
            properties,
        };
        self.image.projection = Some(Projection::Spherical(rep));
        Ok(())
    }

    /// Adds cylindrical image data.
    /// See also `CylindricalImageProperties` struct for more details.
    /// The optional PNG mask image can be used to indicate valid/invalid
    /// pixels in the image, for example if the image is not rectangular.
    /// The mask must have the same size as the actual image.
    /// Non-zero-valued pixels mark valid pixel locations and
    /// zero-valued pixels mark invalid pixels.
    pub fn add_cylindrical(
        &mut self,
        format: ImageFormat,
        image_data: &mut dyn Read,
        properties: CylindricalImageProperties,
        mask_data: Option<&mut dyn Read>,
    ) -> Result<()> {
        if self.image.projection.is_some() {
            Error::invalid("A projected image is already set")?
        }
        let data = Blob::write(self.writer, image_data)?;
        let blob = ImageBlob { data, format };
        let mask = if let Some(mask_data) = mask_data {
            Some(Blob::write(self.writer, mask_data)?)
        } else {
            None
        };
        let rep = CylindricalImage {
            blob,
            mask,
            properties,
        };
        self.image.projection = Some(Projection::Cylindrical(rep));
        Ok(())
    }

    /// Must be called after image is complete to finishing adding the new image.
    /// Binary image and mask data is directly written into the E57 file earlier,
    /// but the XML metadata will be only added to the E57 if you call finalize.
    /// Skipping the finalize call after you added image or mask data means
    /// that the data will be part of the E57 file but is never referenced by
    /// its XML header section.
    pub fn finalize(&mut self) -> Result<()> {
        if self.image.visual_reference.is_none() && self.image.projection.is_none() {
            Error::invalid("Image must have a visual reference or a projection")?
        }

        // Add metadata for XML generation later, when the file is completed.
        self.images.push(self.image.clone());

        Ok(())
    }
}
