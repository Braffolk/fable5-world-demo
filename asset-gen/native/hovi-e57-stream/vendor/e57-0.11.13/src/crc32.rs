/// Simple CRC 32 ISCSI/Castagnoli implementation.
/// This is code is based on the SW fallback of <https://github.com/zowens/crc32c>.
pub struct Crc32 {
    table: [u32; 256],
}

impl Crc32 {
    pub fn new() -> Self {
        let mut table = [0_u32; 256];
        for i in 0..256 {
            let mut val = i;
            for _ in 0..8 {
                if val % 2 == 0 {
                    val /= 2;
                } else {
                    val /= 2;
                    val ^= 0x82_F6_3B_78;
                }
            }
            table[i as usize] = val;
        }
        Self { table }
    }

    pub fn calculate(&mut self, data: &[u8]) -> u32 {
        !data.iter().fold(!0, |sum, &next| {
            let index = (sum ^ next as u32) as u8;
            self.table[index as usize] ^ (sum >> 8)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let data = [0_u8; 0];
        let mut crc = Crc32::new();
        let sum = crc.calculate(&data);
        assert_eq!(sum, 0);
    }

    #[test]
    fn single_u64() {
        let data = [123_u8; 8];
        let mut crc = Crc32::new();
        let sum = crc.calculate(&data);
        assert_eq!(sum, 3786498929);
    }

    #[test]
    fn full_page() {
        let mut data = [0_u8; 1024];
        for i in 0..data.len() {
            data[i] = (i % 256) as u8;
        }
        let mut crc = Crc32::new();
        let sum = crc.calculate(&data);
        assert_eq!(sum, 752840335);
    }
}
