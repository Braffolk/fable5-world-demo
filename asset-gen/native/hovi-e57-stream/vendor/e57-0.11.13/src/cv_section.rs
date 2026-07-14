use crate::error::Converter;
use crate::error::WRONG_OFFSET;
use crate::Error;
use crate::Result;
use std::io::Read;
use std::io::Write;

#[derive(Debug)]
pub struct CompressedVectorSectionHeader {
    section_id: u8,
    pub section_length: u64,
    pub data_offset: u64,
    pub index_offset: u64,
}

impl CompressedVectorSectionHeader {
    pub const SIZE: u64 = 32;

    pub fn read(reader: &mut dyn Read) -> Result<CompressedVectorSectionHeader> {
        let mut buffer = [0_u8; Self::SIZE as usize];
        reader
            .read_exact(&mut buffer)
            .read_err("Failed to read compressed vector section header")?;

        let header = Self {
            section_id: buffer[0],
            section_length: u64::from_le_bytes(
                buffer[8..16].try_into().internal_err(WRONG_OFFSET)?,
            ),
            data_offset: u64::from_le_bytes(buffer[16..24].try_into().internal_err(WRONG_OFFSET)?),
            index_offset: u64::from_le_bytes(buffer[24..32].try_into().internal_err(WRONG_OFFSET)?),
        };

        if header.section_id != 1 {
            Error::invalid("Section ID of the compressed vector section header is not 1")?
        }
        if !header.section_length.is_multiple_of(4) {
            Error::invalid("Section length is not aligned and a multiple of four")?
        }

        Ok(header)
    }

    pub fn write(&self, writer: &mut dyn Write) -> Result<()> {
        let mut buffer = [0_u8; Self::SIZE as usize];
        buffer[0] = self.section_id;
        buffer[8..16].copy_from_slice(&self.section_length.to_le_bytes());
        buffer[16..24].copy_from_slice(&self.data_offset.to_le_bytes());
        buffer[24..32].copy_from_slice(&self.index_offset.to_le_bytes());
        writer
            .write_all(&buffer)
            .write_err("Failed to write compressed vector section header")
    }
}

impl Default for CompressedVectorSectionHeader {
    fn default() -> Self {
        Self {
            section_id: 1,
            section_length: 0,
            data_offset: 0,
            index_offset: 0,
        }
    }
}
