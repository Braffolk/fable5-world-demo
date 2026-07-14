use crate::error::{Converter, WRONG_OFFSET};
use crate::{Error, Result};
use std::io::{Read, Write};

pub enum PacketHeader {
    Index(IndexPacketHeader),
    Data(DataPacketHeader),
    Ignored(IgnoredPacketHeader),
}

impl PacketHeader {
    pub fn read(reader: &mut dyn Read, section_bytes_remaining: u64) -> Result<Self> {
        // Read only first byte of header to indetify packet type
        if section_bytes_remaining == 0 {
            Error::invalid("Compressed vector section ends before the next packet header")?
        }
        let mut buffer = [0_u8; 1];
        reader
            .read_exact(&mut buffer)
            .read_err("Failed to read packet type ID")?;

        if buffer[0] == IndexPacketHeader::ID {
            if section_bytes_remaining < IndexPacketHeader::SIZE {
                Error::invalid("Index packet header crosses compressed vector section boundary")?
            }
            Ok(PacketHeader::Index(IndexPacketHeader::read(reader)?))
        } else if buffer[0] == DataPacketHeader::ID {
            if section_bytes_remaining < DataPacketHeader::SIZE as u64 {
                Error::invalid("Data packet header crosses compressed vector section boundary")?
            }
            Ok(PacketHeader::Data(DataPacketHeader::read(reader)?))
        } else if buffer[0] == IgnoredPacketHeader::ID {
            if section_bytes_remaining < IgnoredPacketHeader::SIZE {
                Error::invalid("Ignored packet header crosses compressed vector section boundary")?
            }
            Ok(PacketHeader::Ignored(IgnoredPacketHeader::read(reader)?))
        } else {
            Error::invalid("Found unknown packet ID when trying to read packet header")?
        }
    }
}

pub struct IndexPacketHeader {
    pub packet_length: u64,
}

impl IndexPacketHeader {
    pub const ID: u8 = 0;
    pub const SIZE: u64 = 16;

    pub fn read(reader: &mut dyn Read) -> Result<Self> {
        let mut buffer = [0_u8; 15];
        reader
            .read_exact(&mut buffer)
            .read_err("Failed to read index packet header")?;

        // Check reserved values in second, eighth, and last eight bytes of header.
        if buffer[0] != 0 || buffer[6] != 0 {
            Error::invalid("The reserved bytes inside an index packet must be zero")?
        }
        for value in buffer.iter().skip(7) {
            if *value != 0 {
                Error::invalid("The reserved bytes inside an index packet must be zero")?
            }
        }

        // Parse values
        let packet_length =
            u16::from_le_bytes(buffer[1..3].try_into().internal_err(WRONG_OFFSET)?) as u64 + 1;

        // Currently unused header fields
        let _entry_count = u16::from_le_bytes(buffer[3..5].try_into().internal_err(WRONG_OFFSET)?);
        let _index_level = buffer[5];

        // Validate length
        if !packet_length.is_multiple_of(4) {
            Error::invalid("Index packet length is not aligned and a multiple of four")?
        }

        Ok(Self { packet_length })
    }
}

pub struct DataPacketHeader {
    pub comp_restart_flag: bool,
    pub packet_length: u64,
    pub bytestream_count: u16,
}

impl DataPacketHeader {
    pub const ID: u8 = 1;

    pub const SIZE: usize = 6;

    pub fn read(reader: &mut dyn Read) -> Result<Self> {
        let mut buffer = [0_u8; 5];
        reader
            .read_exact(&mut buffer)
            .read_err("Failed to read data packet header")?;

        // Parse values
        if buffer[0] & !1 != 0 {
            Error::invalid("Reserved data packet flag bits must be zero")?
        }
        let comp_restart_flag = buffer[0] & 1 != 0;
        let packet_length =
            u16::from_le_bytes(buffer[1..3].try_into().internal_err(WRONG_OFFSET)?) as u64 + 1;
        let bytestream_count =
            u16::from_le_bytes(buffer[3..5].try_into().internal_err(WRONG_OFFSET)?);

        // Validate values
        if !packet_length.is_multiple_of(4) {
            Error::invalid("Data packet length is not aligned and a multiple of four")?
        }
        if bytestream_count == 0 {
            Error::invalid("A byte stream count of 0 is not allowed")?
        }

        Ok(Self {
            comp_restart_flag,
            packet_length,
            bytestream_count,
        })
    }

    pub fn write(&self, writer: &mut dyn Write) -> Result<()> {
        let mut buffer = [0_u8; Self::SIZE];
        buffer[0] = 1;
        let flags = if self.comp_restart_flag { 1_u8 } else { 0_u8 };
        buffer[1] = flags;
        let length = (self.packet_length - 1) as u16;
        buffer[2..4].copy_from_slice(&length.to_le_bytes());
        buffer[4..6].copy_from_slice(&self.bytestream_count.to_le_bytes());
        writer
            .write_all(&buffer)
            .write_err("Failed to write data packet header")
    }
}

pub struct IgnoredPacketHeader {
    pub packet_length: u64,
}

impl IgnoredPacketHeader {
    pub const ID: u8 = 2;
    pub const SIZE: u64 = 4;

    pub fn read(reader: &mut dyn Read) -> Result<Self> {
        // Read Ignored Packet
        let mut buffer = [0_u8; 3];
        reader
            .read_exact(&mut buffer)
            .read_err("Failed to read ignore packet header")?;

        // Check reserved value
        if buffer[0] != 0 {
            Error::invalid("The first byte inside ignored packets is reserved and must be zero")?
        }

        // Parse length
        let packet_length =
            u16::from_le_bytes(buffer[1..3].try_into().internal_err(WRONG_OFFSET)?) as u64 + 1;

        // Validate length
        if !packet_length.is_multiple_of(4) {
            Error::invalid("Ignored packet length is not aligned and a multiple of four")?
        }

        Ok(Self { packet_length })
    }
}
