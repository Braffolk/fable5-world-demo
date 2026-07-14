use crate::bitpack::BitPack;
use crate::bs_read::ByteStreamReadBuffer;
use crate::cv_section::CompressedVectorSectionHeader;
use crate::error::Converter;
use crate::packet::{DataPacketHeader, IgnoredPacketHeader, IndexPacketHeader, PacketHeader};
use crate::paged_reader::PagedReader;
use crate::Error;
use crate::PointCloud;
use crate::RawValues;
use crate::RecordDataType;
use crate::RecordValue;
use crate::Result;
use std::collections::VecDeque;
use std::io::{Read, Seek};

/// Read compressed vector sections into queues of raw values.
pub struct QueueReader<'a, T: Read + Seek> {
    pc: PointCloud,
    reader: &'a mut PagedReader<T>,
    buffer: Vec<u8>,
    buffer_sizes: Vec<usize>,
    byte_streams: Vec<ByteStreamReadBuffer>,
    queues: Vec<VecDeque<RecordValue>>,
    section_end: u64,
    records_popped: u64,
    data_packets_read: u64,
}

impl<'a, T: Read + Seek> QueueReader<'a, T> {
    pub fn new(pc: &PointCloud, reader: &'a mut PagedReader<T>) -> Result<Self> {
        if pc.prototype.is_empty() {
            Error::invalid("Point prototype must contain at least one record")?
        }
        if pc
            .prototype
            .iter()
            .all(|record| record.data_type.bit_size() == 0)
        {
            Error::not_implemented(
                "Point prototypes containing only zero-bit records are not supported",
            )?
        }
        let section_start = reader
            .seek_physical(pc.file_offset)
            .read_err("Cannot seek to compressed vector header")?;
        let section_header = CompressedVectorSectionHeader::read(reader)?;
        if section_header.section_length < CompressedVectorSectionHeader::SIZE {
            Error::invalid("Compressed vector section is shorter than its header")?
        }
        let section_end = section_start
            .checked_add(section_header.section_length)
            .invalid_err("Compressed vector section end overflows u64")?;
        if section_end > reader.logical_size() {
            Error::invalid("Compressed vector section extends beyond the logical file payload")?
        }
        let header_end = section_start + CompressedVectorSectionHeader::SIZE;
        let data_offset = reader
            .logical_offset_for_physical(section_header.data_offset)
            .read_err("Invalid compressed vector data offset")?;
        if data_offset < header_end || data_offset > section_end {
            Error::invalid("Compressed vector data offset is outside its declared section")?
        }
        if pc.records > 0 && data_offset == section_end {
            Error::invalid("Non-empty compressed vector has no data packet region")?
        }
        if section_header.index_offset != 0 {
            let index_offset = reader
                .logical_offset_for_physical(section_header.index_offset)
                .read_err("Invalid compressed vector index offset")?;
            if index_offset < data_offset || index_offset >= section_end {
                Error::invalid("Compressed vector index offset is outside its declared section")?
            }
        }
        reader
            .seek_physical(section_header.data_offset)
            .read_err("Cannot seek to packet header")?;

        Ok(Self {
            pc: pc.clone(),
            reader,
            buffer: Vec::new(),
            buffer_sizes: vec![0; pc.prototype.len()],
            byte_streams: vec![ByteStreamReadBuffer::new(); pc.prototype.len()],
            queues: vec![VecDeque::new(); pc.prototype.len()],
            section_end,
            records_popped: 0,
            data_packets_read: 0,
        })
    }

    /// Returns the number of complete and available points across all queues.
    pub fn available(&self) -> usize {
        if self.queues.is_empty() {
            return 0;
        }

        let mut av = usize::MAX;
        for q in &self.queues {
            let len = q.len();
            if len < av {
                av = len;
            }
        }
        av
    }

    /// Return values for the next point by popping one value from each queue.
    /// Use an existing vector with enough capacity to avoid frequent reallocations!
    pub fn pop_point(&mut self, output: &mut RawValues) -> Result<()> {
        output.clear();
        for i in 0..self.pc.prototype.len() {
            let value = self.queues[i]
                .pop_front()
                .internal_err("Failed to pop value for next point")?;
            output.push(value);
        }
        self.records_popped = self
            .records_popped
            .checked_add(1)
            .internal_err("Decoded record counter overflow")?;
        Ok(())
    }

    /// Reads the next packet from the compressed vector and decodes it into the queues.
    pub fn advance(&mut self) -> Result<()> {
        let packet_start = self.reader.logical_position();
        let section_bytes_remaining = self
            .section_end
            .checked_sub(packet_start)
            .invalid_err("Packet starts beyond compressed vector section boundary")?;
        let packet_header = PacketHeader::read(self.reader, section_bytes_remaining)?;
        let packet_length = match &packet_header {
            PacketHeader::Index(header) => header.packet_length,
            PacketHeader::Data(header) => header.packet_length,
            PacketHeader::Ignored(header) => header.packet_length,
        };
        let packet_end = packet_start
            .checked_add(packet_length)
            .invalid_err("Compressed vector packet end overflows u64")?;
        if packet_end > self.section_end {
            Error::invalid("Compressed vector packet crosses its declared section boundary")?
        }
        match packet_header {
            PacketHeader::Index(header) => {
                let remaining = header
                    .packet_length
                    .checked_sub(IndexPacketHeader::SIZE)
                    .invalid_err("Index packet length is smaller than its header")?;
                skip_exact(self.reader, remaining, "index packet")?
            }
            PacketHeader::Ignored(header) => {
                let remaining = header
                    .packet_length
                    .checked_sub(IgnoredPacketHeader::SIZE)
                    .invalid_err("Ignored packet length is smaller than its header")?;
                skip_exact(self.reader, remaining, "ignored packet")?
            }
            PacketHeader::Data(header) => {
                self.data_packets_read = self
                    .data_packets_read
                    .checked_add(1)
                    .internal_err("Data packet counter overflow")?;
                if header.bytestream_count as usize != self.byte_streams.len() {
                    Error::invalid("Bytestream count does not match prototype size")?
                }
                if header.comp_restart_flag {
                    Error::not_implemented(
                        "Data packets with the compression restart flag are not supported",
                    )?
                }

                // Read byte stream sizes
                for i in 0..self.buffer_sizes.len() {
                    let mut buf = [0_u8; 2];
                    self.reader
                        .read_exact(&mut buf)
                        .read_err("Failed to read data packet buffer sizes")?;
                    let len = u16::from_le_bytes(buf) as usize;
                    self.buffer_sizes[i] = len;
                }

                let size_table_bytes = self
                    .buffer_sizes
                    .len()
                    .checked_mul(std::mem::size_of::<u16>())
                    .internal_err("Data packet size table length overflow")?;
                let stream_bytes = self
                    .buffer_sizes
                    .iter()
                    .try_fold(0_usize, |total, size| total.checked_add(*size))
                    .internal_err("Data packet byte stream length overflow")?;
                let consumed = DataPacketHeader::SIZE
                    .checked_add(size_table_bytes)
                    .and_then(|value| value.checked_add(stream_bytes))
                    .internal_err("Data packet length overflow")?;
                let packet_length = usize::try_from(header.packet_length)
                    .internal_err("Data packet length does not fit in memory")?;
                let padding = packet_length
                    .checked_sub(consumed)
                    .invalid_err("Data packet byte streams exceed declared packet length")?;
                if padding > 3 {
                    Error::invalid(
                        "Data packet declared length has more than three bytes of padding",
                    )?
                }

                // Read byte streams into memory
                for (i, bs) in self.buffer_sizes.iter().enumerate() {
                    self.buffer.resize(*bs, 0_u8);
                    self.reader
                        .read_exact(&mut self.buffer)
                        .read_err("Failed to read data packet buffers")?;
                    self.byte_streams[i].append(&self.buffer);
                }
                skip_exact(self.reader, padding as u64, "data packet padding")?;

                // Find smallest number of expected items in any queue after stream unpacking.
                // This is required for the corner case when the bit size of an record
                // is zero and we don't know how many items to "unpack" from an empty buffer.
                // This happens for example with integer values where min=max, because all values are equal.
                let mut min_queue_size = usize::MAX;
                for (i, bs) in self.byte_streams.iter().enumerate() {
                    let bit_size = self.pc.prototype[i].data_type.bit_size();
                    // We can only check records with a non-zero bit size
                    if let Some(bs_items) = bs.available().checked_div(bit_size) {
                        let queue_items = self.queues[i].len();
                        let items = bs_items
                            .checked_add(queue_items)
                            .internal_err("Decoded queue length overflow")?;
                        if items < min_queue_size {
                            min_queue_size = items;
                        }
                    }
                }

                let records_remaining = self
                    .pc
                    .records
                    .checked_sub(self.records_popped)
                    .internal_err("Decoded more records than the point cloud declares")?;
                let records_remaining = usize::try_from(records_remaining)
                    .internal_err("Remaining record count does not fit in memory")?;
                self.parse_byte_streams(min_queue_size.min(records_remaining))?;
                if self
                    .queues
                    .iter()
                    .any(|queue| queue.len() > records_remaining)
                {
                    Error::invalid(
                        "Compressed vector packet decodes beyond the declared record count",
                    )?
                }
            }
        };

        self.reader
            .align()
            .read_err("Failed to align reader on next 4-byte offset after reading packet")?;
        if self.reader.logical_position() != packet_end {
            Error::invalid("Packet consumption does not match its declared packet length")?
        }
        Ok(())
    }

    /// Consume the rest of the section after the declared records were popped.
    /// Index and ignored packets are allowed; any additional complete record is not.
    pub fn finish(&mut self) -> Result<()> {
        if self.records_popped != self.pc.records {
            Error::invalid("Compressed vector ended before its declared record count")?
        }
        if self.available() != 0 {
            Error::invalid("Decoded queues contain records beyond the declared record count")?
        }
        while self.reader.logical_position() < self.section_end {
            let data_packets_before = self.data_packets_read;
            self.advance()?;
            if self.data_packets_read != data_packets_before {
                Error::invalid("Compressed vector has a data packet after its declared records")?
            }
            if self.available() != 0 {
                Error::invalid("Compressed vector contains records beyond its declared count")?
            }
        }
        if self.reader.logical_position() != self.section_end {
            Error::invalid("Compressed vector reader did not finish at the section boundary")?
        }
        Ok(())
    }

    /// Extracts raw values from byte streams into queues.
    fn parse_byte_streams(&mut self, target_queue_size: usize) -> Result<()> {
        for (i, r) in self.pc.prototype.iter().enumerate() {
            let max_items = target_queue_size.saturating_sub(self.queues[i].len());
            match r.data_type {
                RecordDataType::Single { .. } => BitPack::unpack_singles(
                    &mut self.byte_streams[i],
                    &mut self.queues[i],
                    max_items,
                )?,
                RecordDataType::Double { .. } => BitPack::unpack_doubles(
                    &mut self.byte_streams[i],
                    &mut self.queues[i],
                    max_items,
                )?,
                RecordDataType::ScaledInteger { min, max, .. } => {
                    if r.data_type.bit_size() == 0 {
                        // If the bit size of an record is zero, we don't know how many items to unpack.
                        // Thats because they are not really unpacked, but instead generated with a predefined value.
                        // Since this can only happen when min=max we know that min is the expected value.
                        // We use the supplied minimal size to ensure that we create enough items
                        // to fill the queue enough to not be the limiting queue.
                        while self.queues[i].len() < target_queue_size {
                            self.queues[i].push_back(RecordValue::ScaledInteger(min));
                        }
                    } else {
                        BitPack::unpack_scaled_ints(
                            &mut self.byte_streams[i],
                            min,
                            max,
                            &mut self.queues[i],
                            max_items,
                        )?
                    }
                }
                RecordDataType::Integer { min, max } => {
                    if r.data_type.bit_size() == 0 {
                        // See comment above for scaled integers!
                        while self.queues[i].len() < target_queue_size {
                            self.queues[i].push_back(RecordValue::Integer(min));
                        }
                    } else {
                        BitPack::unpack_ints(
                            &mut self.byte_streams[i],
                            min,
                            max,
                            &mut self.queues[i],
                            max_items,
                        )?
                    }
                }
            };
        }

        if self
            .queues
            .iter()
            .any(|queue| queue.len() != target_queue_size)
        {
            Error::invalid("Compressed-vector bytestreams did not produce aligned record queues")?
        }

        Ok(())
    }
}

fn skip_exact<T: Read + Seek>(
    reader: &mut PagedReader<T>,
    mut bytes: u64,
    label: &str,
) -> Result<()> {
    let mut buffer = [0_u8; 4096];
    while bytes > 0 {
        let count = usize::try_from(bytes.min(buffer.len() as u64))
            .internal_err("Packet skip length does not fit in memory")?;
        reader
            .read_exact(&mut buffer[..count])
            .read_err(format!("Failed to read remaining data of {label}"))?;
        bytes -= count as u64;
    }
    Ok(())
}
