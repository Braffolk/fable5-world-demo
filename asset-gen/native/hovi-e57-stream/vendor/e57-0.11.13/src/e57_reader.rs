use crate::error::Converter;
use crate::paged_reader::PagedReader;
use crate::root::root_from_document;
use crate::root::Root;
use crate::Blob;
use crate::DateTime;
use crate::Error;
use crate::Extension;
use crate::Header;
use crate::Image;
use crate::PointCloud;
use crate::PointCloudReaderRaw;
use crate::PointCloudReaderSimple;
use crate::Result;
use roxmltree::Document;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;
use std::io::Write;
use std::path::Path;

const MAX_XML_SIZE: usize = 1024 * 1024 * 50;

/// Main interface for reading E57 files.
pub struct E57Reader<T: Read + Seek> {
    reader: PagedReader<T>,
    header: Header,
    xml: String,
    root: Root,
    pointclouds: Vec<PointCloud>,
    images: Vec<Image>,
    extensions: Vec<Extension>,
}

impl<T: Read + Seek> E57Reader<T> {
    /// Creates a new E57 instance for from a reader.
    pub fn new(mut reader: T) -> Result<Self> {
        // Read, parse and validate E57 header
        let header = Header::read(&mut reader)?;

        // Set up paged reader for the CRC page layer
        let mut reader = PagedReader::new(reader, header.page_size)
            .read_err("Failed creating paged CRC reader")?;

        // Read and parse XML data
        let xml_raw = Self::extract_xml(
            &mut reader,
            header.phys_xml_offset,
            header.xml_length as usize,
        )?;
        let xml = String::from_utf8(xml_raw).read_err("Failed to parse XML as UTF8")?;
        let document = Document::parse(&xml).invalid_err("Failed to parse XML data")?;
        let root = root_from_document(&document)?;
        let pointclouds = PointCloud::vec_from_document(&document)?;
        let images = Image::vec_from_document(&document)?;
        let extensions = Extension::vec_from_document(&document);

        Ok(Self {
            reader,
            header,
            xml,
            root,
            pointclouds,
            images,
            extensions,
        })
    }

    /// Returns the contents of E57 binary file header structure.
    pub fn header(&self) -> Header {
        self.header.clone()
    }

    /// Returns the XML section of the E57 file.
    pub fn xml(&self) -> &str {
        &self.xml
    }

    /// Returns format name stored in the XML section.
    pub fn format_name(&self) -> &str {
        &self.root.format
    }

    /// Returns GUID stored in the XML section.
    pub fn guid(&self) -> &str {
        &self.root.guid
    }

    /// Returns the library version string of the root XML section.
    pub fn library_version(&self) -> Option<&str> {
        self.root.library_version.as_deref()
    }

    /// Returns a list of all extensions defined in this file.
    pub fn extensions(&self) -> Vec<Extension> {
        self.extensions.clone()
    }

    /// Returns a list of all point cloud descriptors in the file.
    pub fn pointclouds(&self) -> Vec<PointCloud> {
        self.pointclouds.clone()
    }

    /// Returns an iterator for reading point cloud data.
    /// The data provided by this interface is already normalized for convenience.
    /// There is also a raw iterator for advanced use-cases that require direct access.
    pub fn pointcloud_simple(&mut self, pc: &PointCloud) -> Result<PointCloudReaderSimple<'_, T>> {
        PointCloudReaderSimple::new(pc, &mut self.reader)
    }

    /// Returns an iterator for reading raw low level point cloud data.
    /// This provides access to the original values stored in the E57 file.
    /// This interface is only recommended for advanced use-cases.
    /// In most scenarios the simple iterator is the better choice.
    pub fn pointcloud_raw(&mut self, pc: &PointCloud) -> Result<PointCloudReaderRaw<'_, T>> {
        PointCloudReaderRaw::new(pc, &mut self.reader)
    }

    /// Returns a list of all image descriptors in the file.
    pub fn images(&self) -> Vec<Image> {
        self.images.clone()
    }

    /// Reads the content of a blob and copies it into the supplied writer.
    /// Returns the number of written bytes.
    pub fn blob(&mut self, blob: &Blob, writer: &mut dyn Write) -> Result<u64> {
        blob.read(&mut self.reader, writer)
    }

    /// Returns the optional creation date and time of the file.
    pub fn creation(&self) -> Option<DateTime> {
        self.root.creation.clone()
    }

    /// Returns the optional coordinate system metadata of the file.
    ///
    /// This should contain a Coordinate Reference System that is specified by
    /// a string in a well-known text format for a spatial reference system,
    /// as defined by the Coordinate Transformation Service specification
    /// developed by the Open Geospatial Consortium.
    /// See also: <https://www.ogc.org/standard/wkt-crs/>
    pub fn coordinate_metadata(&self) -> Option<&str> {
        self.root.coordinate_metadata.as_deref()
    }

    /// Iterate over an reader to check an E57 file for CRC errors.
    ///
    /// This standalone function does only the minimal parsing required
    /// to get the E57 page size and without any other checks or validation.
    /// After that it will CRC-validate the whole file.
    /// It will not read or check any other file header and XML data!
    /// This method returns the page size of the E57 file.
    pub fn validate_crc(mut reader: T) -> Result<u64> {
        let page_size = Self::get_u64(&mut reader, 40, "page size")?;
        let mut paged_reader =
            PagedReader::new(reader, page_size).read_err("Failed creating paged CRC reader")?;
        let mut buffer = vec![0_u8; page_size as usize];
        let mut page = 0;
        while paged_reader
            .read(&mut buffer)
            .read_err(format!("Failed to validate CRC for page {page}"))?
            != 0
        {
            page += 1;
        }
        Ok(page_size)
    }

    /// Returns the raw unparsed binary XML data of the E57 file as bytes.
    ///
    /// This standalone function does only the minimal parsing required
    /// to get the XML section without any other checks or any other
    /// validation than basic CRC ckecking for the XML section itself.
    pub fn raw_xml(mut reader: T) -> Result<Vec<u8>> {
        let page_size = Self::get_u64(&mut reader, 40, "page size")?;
        let xml_offset = Self::get_u64(&mut reader, 24, "XML offset")?;
        let xml_length = Self::get_u64(&mut reader, 32, "XML length")?;

        // Create paged CRC reader
        let mut paged_reader =
            PagedReader::new(reader, page_size).read_err("Failed creating paged CRC reader")?;

        // Read XML data
        Self::extract_xml(&mut paged_reader, xml_offset, xml_length as usize)
    }

    fn get_u64(reader: &mut T, offset: u64, name: &str) -> Result<u64> {
        reader
            .seek(std::io::SeekFrom::Start(offset))
            .read_err(format!("Cannot seek to {name} offset"))?;
        let mut buf = [0_u8; 8];
        reader
            .read_exact(&mut buf)
            .read_err(format!("Cannot read {name} bytes"))?;
        Ok(u64::from_le_bytes(buf))
    }

    fn extract_xml(reader: &mut PagedReader<T>, offset: u64, length: usize) -> Result<Vec<u8>> {
        if length > MAX_XML_SIZE {
            Error::not_implemented(format!(
                "XML sections larger than {MAX_XML_SIZE} bytes are not supported"
            ))?
        }
        reader.seek_physical(offset).read_err(format!(
            "File appears to be truncated: cannot seek to XML section at offset {offset}"
        ))?;
        let mut xml = vec![0_u8; length];
        reader.read_exact(&mut xml).read_err(format!(
            "File appears to be truncated: cannot read {length} bytes of XML data"
        ))?;
        Ok(xml)
    }
}

impl E57Reader<BufReader<File>> {
    /// Creates an E57 instance from a Path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).read_err("Unable to open file")?;
        let reader = BufReader::new(file);
        Self::new(reader)
    }
}
