use crate::error::Converter;
use crate::error::WRONG_OFFSET;
use crate::Error;
use crate::Result;
use std::io::Read;
use std::io::Write;

const SIGNATURE: &[u8; 8] = b"ASTM-E57";
const MAJOR_VERSION: u32 = 1;
const MINOR_VERSION: u32 = 0;
const PAGE_SIZE: u64 = 1024;

/// Represents the file structure from the start of an E57 file.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Header {
    /// File header signature that must be always "ASTM-E57".
    pub signature: [u8; 8],

    /// Major version number of the E57 format of the file.
    pub major: u32,

    /// Minor version number of the E57 format of the file.
    pub minor: u32,

    /// Physical length of the E57 file on disk or in memory.
    pub phys_length: u64,

    /// Physical offset of the XML data inside the XML file.
    pub phys_xml_offset: u64,

    /// Logical (without CRC bytes) length of the XML data.
    pub xml_length: u64,

    /// Page size of the E57 file.
    pub page_size: u64,
}

impl Header {
    /// Reads an E57 file header structure.
    pub fn read(reader: &mut dyn Read) -> Result<Self> {
        let mut data = [0_u8; 48];
        reader
            .read_exact(&mut data)
            .read_err("Failed to read E57 file header")?;

        let header = Header {
            signature: data[0..8].try_into().internal_err(WRONG_OFFSET)?,
            major: u32::from_le_bytes(data[8..12].try_into().internal_err(WRONG_OFFSET)?),
            minor: u32::from_le_bytes(data[12..16].try_into().internal_err(WRONG_OFFSET)?),
            phys_length: u64::from_le_bytes(data[16..24].try_into().internal_err(WRONG_OFFSET)?),
            phys_xml_offset: u64::from_le_bytes(
                data[24..32].try_into().internal_err(WRONG_OFFSET)?,
            ),
            xml_length: u64::from_le_bytes(data[32..40].try_into().internal_err(WRONG_OFFSET)?),
            page_size: u64::from_le_bytes(data[40..48].try_into().internal_err(WRONG_OFFSET)?),
        };

        if &header.signature != SIGNATURE {
            Error::invalid("Found unsupported signature in header")?
        }
        if header.major != MAJOR_VERSION {
            Error::invalid("Found unsupported major version in header")?
        }
        if header.minor != MINOR_VERSION {
            Error::invalid("Found unsupported minor version in header")?
        }
        if header.page_size != PAGE_SIZE {
            Error::invalid("Found unsupported page size in header")?
        }

        Ok(header)
    }

    pub fn write(&self, writer: &mut dyn Write) -> Result<()> {
        writer
            .write_all(&self.signature)
            .write_err("Failed to write file header signature")?;
        writer
            .write_all(&self.major.to_le_bytes())
            .write_err("Failed to write file header major version")?;
        writer
            .write_all(&self.minor.to_le_bytes())
            .write_err("Failed to write file header minor version")?;
        writer
            .write_all(&self.phys_length.to_le_bytes())
            .write_err("Failed to write file length in file header")?;
        writer
            .write_all(&self.phys_xml_offset.to_le_bytes())
            .write_err("Failed to write XML offset in file header")?;
        writer
            .write_all(&self.xml_length.to_le_bytes())
            .write_err("Failed to write XML length in file header")?;
        writer
            .write_all(&self.page_size.to_le_bytes())
            .write_err("Failed to write page size in file header")?;
        Ok(())
    }
}

impl Default for Header {
    fn default() -> Self {
        Self {
            signature: *SIGNATURE,
            major: MAJOR_VERSION,
            minor: MINOR_VERSION,
            phys_length: 0,
            phys_xml_offset: 0,
            xml_length: 0,
            page_size: PAGE_SIZE,
        }
    }
}
