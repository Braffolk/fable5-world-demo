use crate::error::Converter;
use crate::paged_writer::PagedWriter;
use crate::pc_writer::PointCloudWriter;
use crate::root::{serialize_root, Root};
use crate::{
    Blob, DateTime, Error, Extension, Header, Image, ImageWriter, PointCloud, Record, Result,
};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::Path;

/// Main interface for creating and writing E57 files.
pub struct E57Writer<T: Read + Write + Seek> {
    pub(crate) writer: PagedWriter<T>,
    pub(crate) pointclouds: Vec<PointCloud>,
    extensions: Vec<Extension>,
    images: Vec<Image>,
    root: Root,
}

impl<T: Write + Read + Seek> E57Writer<T> {
    /// Creates a new E57 generator from a writer that must also implement Read and Seek.
    ///
    /// `File::create()` will not work as input because it only opens the file for writing.
    /// Most typical use cases should prefer `E57Writer::from_file()` over this constructor.
    pub fn new(writer: T, guid: &str) -> Result<Self> {
        // Set up paged writer abstraction for CRC
        let mut writer = PagedWriter::new(writer)?;

        // Write placeholder header that will be replaced later
        let header = Header::default();
        header.write(&mut writer)?;

        let version = env!("CARGO_PKG_VERSION");
        let library_version = Some(format!(
            "Rust E57 Library v{version} github.com/cry-inc/e57"
        ));
        let root = Root {
            guid: guid.to_owned(),
            library_version,
            ..Default::default()
        };

        Ok(Self {
            writer,
            pointclouds: Vec::new(),
            images: Vec::new(),
            extensions: Vec::new(),
            root,
        })
    }

    /// Set optional coordinate metadata string (empty by default).
    pub fn set_coordinate_metadata(&mut self, value: Option<String>) {
        self.root.coordinate_metadata = value;
    }

    /// Set optional creation date time (empty by default).
    pub fn set_creation(&mut self, value: Option<DateTime>) {
        self.root.creation = value;
    }

    /// Creates a new writer for adding a new point cloud to the E57 file.
    pub fn add_pointcloud(
        &mut self,
        guid: &str,
        prototype: Vec<Record>,
    ) -> Result<PointCloudWriter<'_, T>> {
        Extension::validate_prototype(&prototype, &self.extensions)?;
        PointCloudWriter::new(&mut self.writer, &mut self.pointclouds, guid, prototype)
    }

    /// Adds a new binary data section to the E57 file.
    /// This feature is only required for custom data and extensions!
    pub fn add_blob(&mut self, reader: &mut dyn Read) -> Result<Blob> {
        Blob::write(&mut self.writer, reader)
    }

    /// Creates a new image writer for adding an image to the E57 file.
    pub fn add_image(&mut self, guid: &str) -> Result<ImageWriter<'_, T>> {
        ImageWriter::new(&mut self.writer, &mut self.images, guid)
    }

    /// Registers a new E57 extension used by this file.
    pub fn register_extension(&mut self, extension: Extension) -> Result<()> {
        Extension::validate_name(&extension.namespace)?;
        if self
            .extensions
            .iter()
            .any(|e| e.namespace == extension.namespace)
        {
            let ns = &extension.namespace;
            Error::invalid(format!(
                "An extension using the namespace {ns} is already registered"
            ))?
        } else {
            self.extensions.push(extension);
            Ok(())
        }
    }

    /// Needs to be called after adding all point clouds and images.
    ///
    /// This will generate and write the XML metadata to finalize and complete the E57 file.
    /// Without calling this method before dropping the E57 file will be incomplete and invalid!
    pub fn finalize(&mut self) -> Result<()> {
        self.finalize_customized_xml(Ok)
    }

    /// Same as `finalize()` but with additional XML transformation step.
    ///
    /// Allows customizing the XML data before its written into the E57 file.
    /// This is required for adding E57 extension data to the XML.
    /// The transformer receives an XML string and must return an XML string.
    /// The client is responsible for parsing, modifying and serializing th XML again in a non-destructive way.
    /// The E57 library will not validate the XML string before writing it into the E57 file!
    /// If the transformer fails, the finalization is aborted and any error is forwarded.
    pub fn finalize_customized_xml(
        &mut self,
        transformer: impl Fn(String) -> Result<String>,
    ) -> Result<()> {
        let xml = serialize_root(
            &self.root,
            &self.pointclouds,
            &self.images,
            &self.extensions,
        )?;
        let xml = transformer(xml)?;
        let xml_bytes = xml.as_bytes();
        let xml_length = xml_bytes.len();
        let xml_offset = self.writer.physical_position()?;
        self.writer
            .write_all(xml_bytes)
            .write_err("Failed to write XML data")?;
        let phys_length = self.writer.physical_size()?;

        // Add missing values in header at start of the the file
        let header = Header {
            phys_xml_offset: xml_offset,
            xml_length: xml_length as u64,
            phys_length,
            ..Default::default()
        };
        self.writer.physical_seek(0)?;
        header.write(&mut self.writer)?;
        self.writer
            .flush()
            .write_err("Failed to flush writer at the end")
    }
}

impl E57Writer<File> {
    /// Creates an E57 writer instance from a Path.
    pub fn from_file(path: impl AsRef<Path>, guid: &str) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(path)
            .read_err("Unable to create file for writing, reading and seeking")?;
        Self::new(file, guid)
    }
}
