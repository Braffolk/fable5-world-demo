#[derive(Clone)]
pub struct ByteStreamReadBuffer {
    buffer: Vec<u8>,
    tmp: Vec<u8>,
    offset: usize,
}

impl ByteStreamReadBuffer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            tmp: Vec::new(),
            offset: 0,
        }
    }

    /// Append a fresh slice of bytes to the end of the stream
    pub fn append(&mut self, data: &[u8]) {
        let consumed_bytes = self.offset / 8;
        let remaining_bytes = self.buffer.len() - consumed_bytes;
        self.offset -= consumed_bytes * 8;
        self.tmp.reserve(remaining_bytes + data.len());
        self.tmp.extend_from_slice(&self.buffer[consumed_bytes..]);
        self.tmp.extend_from_slice(data);
        self.buffer.clear();
        std::mem::swap(&mut self.buffer, &mut self.tmp);
    }

    /// Extract 64 bits or less from the byte stream and return them as u64.
    /// The returned u64 might contain more than the requested number of bits.
    /// Please make sure to ignore/mask the additional bits!
    /// Returns None if the request cannot be satisfied.
    pub fn extract(&mut self, bits: usize) -> Option<u64> {
        if self.available() < bits {
            return None;
        }

        let start_offset = self.offset / 8;
        let end_offset = (self.offset + bits).div_ceil(8); // Integer division with rounding up
        let offset = self.offset % 8;

        let mut data = [0; 16];
        let data_len = end_offset - start_offset;
        let dst = &mut data[..data_len];
        let src = &self.buffer[start_offset..end_offset];
        dst.copy_from_slice(src);

        self.offset += bits;
        let data = u128::from_le_bytes(data) >> offset;
        Some(data as u64)
    }

    /// Returns the number of available bits in the stream
    pub fn available(&self) -> usize {
        (self.buffer.len() * 8) - self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let mut bs = ByteStreamReadBuffer::new();
        assert_eq!(bs.available(), 0);
        let result = bs.extract(0).unwrap();
        assert_eq!(result, 0);
        assert_eq!(bs.available(), 0);
        assert!(bs.extract(1).is_none());
    }

    #[test]
    fn append_and_extract_bits() {
        let mut bs = ByteStreamReadBuffer::new();
        bs.append(&[255]);

        assert_eq!(bs.available(), 8);
        let result = bs.extract(2).unwrap();
        assert_eq!(result, 255);

        assert_eq!(bs.available(), 6);
        let result = bs.extract(6).unwrap();
        assert_eq!(result, 63);

        assert_eq!(bs.available(), 0);
        assert!(bs.extract(1).is_none());
    }

    #[test]
    fn append_and_extract_bytes() {
        let mut bs = ByteStreamReadBuffer::new();
        bs.append(&[23, 42, 13]);
        bs.extract(2).unwrap();

        assert_eq!(bs.available(), 22);
        let result = bs.extract(22).unwrap();
        assert_eq!(result, 215685);
    }

    #[test]
    fn remove_consume_when_appending() {
        let mut bs = ByteStreamReadBuffer::new();
        bs.append(&[1, 2, 3, 4, 5]);
        bs.extract(4 * 8 + 2).unwrap();

        // We append one byte and the buffer should become smaller
        // because all fully consumed bytes are removed.
        bs.append(&[6]);
        assert!(bs.buffer.len() == 2);

        // Offsets are updated correctly appended
        // data can be extracted as expected.
        let result = bs.extract(14).unwrap();
        assert_eq!(result, 385);
    }
}
