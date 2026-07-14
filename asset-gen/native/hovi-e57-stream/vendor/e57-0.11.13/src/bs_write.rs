#[derive(Clone)]
pub struct ByteStreamWriteBuffer {
    buffer: Vec<u8>,
    last_byte_bit: usize,
}

impl ByteStreamWriteBuffer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            last_byte_bit: 0,
        }
    }

    pub fn add_bytes(&mut self, data: &[u8]) {
        if self.last_byte_bit == 0 {
            self.buffer.extend_from_slice(data);
        } else {
            self.add_bits(data, data.len() * 8)
        }
    }

    pub fn add_bits(&mut self, data: &[u8], bits: usize) {
        if self.last_byte_bit == 0 {
            let to_append = bits.div_ceil(8); // Integer division with rounding up
            self.buffer.extend_from_slice(&data[..to_append]);
            self.last_byte_bit = bits % 8;
        } else {
            let start_byte = self.buffer.len() - 1;
            let start_bit = self.last_byte_bit;
            for b in 0..bits {
                let source_byte = b / 8;
                let source_mask = 1 << (b % 8);
                let source_bit = (data[source_byte] & source_mask) != 0;
                let target_mask = if source_bit {
                    1 << self.last_byte_bit
                } else {
                    0
                };
                let target_byte = start_byte + ((start_bit + b) / 8);
                if target_byte >= self.buffer.len() {
                    self.buffer.push(0);
                }
                self.buffer[target_byte] |= target_mask;
                self.last_byte_bit = (self.last_byte_bit + 1) % 8;
            }
        }
    }

    pub fn get_full_bytes(&mut self) -> Vec<u8> {
        let to_take = self.full_bytes();
        self.buffer.drain(..to_take).collect()
    }

    pub fn get_all_bytes(&mut self) -> Vec<u8> {
        self.last_byte_bit = 0;
        self.buffer.drain(..).collect()
    }

    pub fn full_bytes(&self) -> usize {
        let len = self.buffer.len();
        if self.last_byte_bit != 0 {
            len - 1
        } else {
            len
        }
    }

    pub fn all_bytes(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let mut buffer = ByteStreamWriteBuffer::new();
        assert_eq!(buffer.full_bytes(), 0);
        assert_eq!(buffer.all_bytes(), 0);

        let full = buffer.get_full_bytes();
        assert_eq!(full.len(), 0);

        let all = buffer.get_all_bytes();
        assert_eq!(all.len(), 0);
    }

    #[test]
    fn add_bytes() {
        let mut buffer = ByteStreamWriteBuffer::new();
        buffer.add_bytes(&[1, 2, 3, 4]);
        assert_eq!(buffer.full_bytes(), 4);
        assert_eq!(buffer.all_bytes(), 4);

        let full = buffer.get_full_bytes();
        assert_eq!(full.len(), 4);
        assert_eq!(full, [1, 2, 3, 4]);

        assert_eq!(buffer.full_bytes(), 0);
        assert_eq!(buffer.all_bytes(), 0);
    }

    #[test]
    fn add_bits_after_full_bytes() {
        let mut buffer = ByteStreamWriteBuffer::new();
        buffer.add_bytes(&[1, 2, 3, 4]);
        buffer.add_bits(&[0b00001111], 4);

        assert_eq!(buffer.full_bytes(), 4);
        assert_eq!(buffer.all_bytes(), 5);

        let full = buffer.get_full_bytes();
        assert_eq!(full.len(), 4);
        assert_eq!(full, [1, 2, 3, 4]);
        assert_eq!(buffer.full_bytes(), 0);
        assert_eq!(buffer.all_bytes(), 1);

        let all = buffer.get_all_bytes();
        assert_eq!(all.len(), 1);
        assert_eq!(all, [0b00001111]);

        assert_eq!(buffer.full_bytes(), 0);
        assert_eq!(buffer.all_bytes(), 0);
    }

    #[test]
    fn add_two_times_four_bits() {
        let mut buffer = ByteStreamWriteBuffer::new();
        buffer.add_bits(&[0b00001111], 4);
        buffer.add_bits(&[0b00001111], 4);

        assert_eq!(buffer.full_bytes(), 1);
        assert_eq!(buffer.all_bytes(), 1);

        let all = buffer.get_all_bytes();
        assert_eq!(all, [0b11111111]);
    }

    #[test]
    fn add_mixed_bits_and_bytes() {
        let mut buffer = ByteStreamWriteBuffer::new();
        buffer.add_bits(&[0b101], 3);
        buffer.add_bytes(&[0b10000001]);
        buffer.add_bits(&[0b100001], 6);

        assert_eq!(buffer.full_bytes(), 2);
        assert_eq!(buffer.all_bytes(), 3);

        let all = buffer.get_all_bytes();
        assert_eq!(all, [0b00001101, 0b00001100, 0b00000001]);
    }
}
