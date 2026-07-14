use crate::error::{Converter, WRONG_OFFSET};
use crate::paged_reader::PagedReader;
use crate::paged_writer::PagedWriter;
use crate::{Error, Result};
use roxmltree::Node;
use std::io::{copy, Read, Seek, Write};

/// Describes a binary data blob stored inside an E57 file.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Blob {
    /// Physical file offset of the binary blob section in the E57 file.
    pub offset: u64,
    /// The logical size of the associated binary blob in bytes.
    pub length: u64,
}

impl Blob {
    /// Creates a blob instance manually from offset and length.
    /// WARNING: This constructor is NOT required for standard E57 functionality!
    /// In normal cases, like when reading images, the library will provide all Blob instances.
    /// This is only needed for use cases with E57 extensions to read custom binary data.
    /// In this case the offset and length values must be extracted manually from the XML data.
    pub fn new(offset: u64, length: u64) -> Self {
        Blob { offset, length }
    }

    pub(crate) fn from_node(node: &Node) -> Result<Self> {
        if Some("Blob") != node.attribute("type") {
            Error::invalid("The supplided tag is not a blob")?
        }

        let offset = node
            .attribute("fileOffset")
            .invalid_err("Failed to find 'fileOffset' attribute in blob tag")?;
        let offset = offset
            .parse::<u64>()
            .invalid_err("Unable to parse offset as u64")?;

        let length = node
            .attribute("length")
            .invalid_err("Failed to find 'length' attribute in blob tag")?;
        let length = length
            .parse::<u64>()
            .invalid_err("Unable to parse length as u64")?;

        Ok(Self { offset, length })
    }

    pub(crate) fn from_parent_node(tag_name: &str, parent_node: &Node) -> Result<Option<Self>> {
        if let Some(node) = &parent_node.children().find(|n| n.has_tag_name(tag_name)) {
            Ok(Some(Self::from_node(node)?))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn xml_string(&self, tag_name: &str) -> String {
        format!(
            "<{tag_name} type=\"Blob\" fileOffset=\"{}\" length=\"{}\"/>\n",
            self.offset, self.length
        )
    }

    pub(crate) fn read<T: Read + Seek>(
        &self,
        reader: &mut PagedReader<T>,
        writer: &mut dyn Write,
    ) -> Result<u64> {
        reader
            .seek_physical(self.offset)
            .read_err("Failed to seek to start offset of blob")?;
        let header = BlobSectionHeader::from_reader(reader)?;
        if self.length > header.section_length + 16 {
            Error::invalid("Blob XML length and blob section header mismatch")?
        }

        let mut limited = reader.take(self.length);
        copy(&mut limited, writer).read_err("Failed to read binary blob data")
    }

    pub(crate) fn write<T: Read + Write + Seek>(
        writer: &mut PagedWriter<T>,
        reader: &mut dyn Read,
    ) -> Result<Self> {
        // Write temporary section header with invalid zero length
        let start_offset = writer.physical_position()?;
        let mut section_header = BlobSectionHeader { section_length: 0 };
        section_header.to_writer(writer)?;

        // Write blob data
        let length = std::io::copy(reader, writer).write_err("Failed to write blob data")?;

        // Update blob section header with actual lenght
        let end_offset = writer.physical_position()?;
        section_header.section_length = length;
        writer.physical_seek(start_offset)?;
        section_header.to_writer(writer)?;
        writer.physical_seek(end_offset)?;

        writer
            .align()
            .write_err("Failed to align writer on next 4-byte offset after writing blob section")?;

        Ok(Self {
            offset: start_offset,
            length,
        })
    }
}

struct BlobSectionHeader {
    section_length: u64,
}

impl BlobSectionHeader {
    fn from_array(buffer: &[u8; 16]) -> Result<Self> {
        let section_id = buffer[0];
        if section_id != 0 {
            Error::invalid("Section ID of the blob section header is not 0")?
        }
        Ok(Self {
            section_length: u64::from_le_bytes(
                buffer[8..16].try_into().internal_err(WRONG_OFFSET)?,
            ),
        })
    }

    fn from_reader<T: Read + Seek>(reader: &mut PagedReader<T>) -> Result<BlobSectionHeader> {
        let mut buffer = [0_u8; 16];
        reader
            .read_exact(&mut buffer)
            .read_err("Failed to read compressed vector section header")?;
        BlobSectionHeader::from_array(&buffer)
    }

    fn to_writer<T: Read + Write + Seek>(&self, writer: &mut PagedWriter<T>) -> Result<()> {
        let mut bytes: [u8; 16] = [0; 16];
        let length_bytes = u64::to_le_bytes(self.section_length);
        bytes[8..16].copy_from_slice(&length_bytes);
        writer
            .write_all(&bytes)
            .write_err("Failed to write blob section header")
    }
}
