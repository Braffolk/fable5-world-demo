use crate::bs_read::ByteStreamReadBuffer;
use crate::Error;
use crate::RecordValue;
use crate::Result;
use std::collections::VecDeque;

pub struct BitPack;

impl BitPack {
    pub fn unpack_doubles(
        stream: &mut ByteStreamReadBuffer,
        output: &mut VecDeque<RecordValue>,
        max_items: usize,
    ) -> Result<()> {
        for _ in 0..max_items {
            let Some(data) = stream.extract(64) else {
                break;
            };
            let bytes = data.to_le_bytes();
            let value = f64::from_le_bytes(bytes);
            output.push_back(RecordValue::Double(value));
        }
        Ok(())
    }

    pub fn unpack_singles(
        stream: &mut ByteStreamReadBuffer,
        output: &mut VecDeque<RecordValue>,
        max_items: usize,
    ) -> Result<()> {
        for _ in 0..max_items {
            let Some(data) = stream.extract(32) else {
                break;
            };
            let bytes = (data as u32).to_le_bytes();
            let value = f32::from_le_bytes(bytes);
            output.push_back(RecordValue::Single(value));
        }
        Ok(())
    }

    pub fn unpack_ints(
        stream: &mut ByteStreamReadBuffer,
        min: i64,
        max: i64,
        output: &mut VecDeque<RecordValue>,
        max_items: usize,
    ) -> Result<()> {
        let (range, bits, mask) = integer_layout(min, max)?;
        if bits == 0 {
            return Ok(());
        }
        for _ in 0..max_items {
            let Some(uint) = stream.extract(bits) else {
                break;
            };
            let code = uint & mask;
            if i128::from(code) > range {
                Error::invalid("Integer bit code exceeds the prototype's declared maximum")?
            }
            let int = i128::from(code) + min as i128;
            output.push_back(RecordValue::Integer(int as i64));
        }
        Ok(())
    }

    pub fn unpack_scaled_ints(
        stream: &mut ByteStreamReadBuffer,
        min: i64,
        max: i64,
        output: &mut VecDeque<RecordValue>,
        max_items: usize,
    ) -> Result<()> {
        let (range, bits, mask) = integer_layout(min, max)?;
        if bits == 0 {
            return Ok(());
        }
        for _ in 0..max_items {
            let Some(uint) = stream.extract(bits) else {
                break;
            };
            let code = uint & mask;
            if i128::from(code) > range {
                Error::invalid("Scaled-integer bit code exceeds the prototype's declared maximum")?
            }
            let int = i128::from(code) + min as i128;
            output.push_back(RecordValue::ScaledInteger(int as i64));
        }
        Ok(())
    }
}

fn integer_layout(min: i64, max: i64) -> Result<(i128, usize, u64)> {
    if max < min {
        Error::invalid("Integer prototype maximum is smaller than its minimum")?
    }
    let range = max as i128 - min as i128;
    if range == 0 {
        return Ok((range, 0, 0));
    }
    let bits = range.ilog2() as usize + 1;
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    };
    Ok((range, bits, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_neighboring_bits_before_validating_integer_codes() {
        let mut stream = ByteStreamReadBuffer::new();
        // Two 3-bit codes (1, 2), followed by two packet-fragment bits.
        stream.append(&[0b1101_0001]);
        let mut output = VecDeque::new();

        BitPack::unpack_ints(&mut stream, -2, 3, &mut output, usize::MAX).unwrap();

        assert_eq!(
            output,
            VecDeque::from([RecordValue::Integer(-1), RecordValue::Integer(0)])
        );
        assert_eq!(stream.available(), 2);
    }

    #[test]
    fn masks_neighboring_bits_before_validating_scaled_integer_codes() {
        let mut stream = ByteStreamReadBuffer::new();
        stream.append(&[0b1101_0001]);
        let mut output = VecDeque::new();

        BitPack::unpack_scaled_ints(&mut stream, -2, 3, &mut output, usize::MAX).unwrap();

        assert_eq!(
            output,
            VecDeque::from([
                RecordValue::ScaledInteger(-1),
                RecordValue::ScaledInteger(0),
            ])
        );
        assert_eq!(stream.available(), 2);
    }

    #[test]
    fn rejects_unused_code_after_masking() {
        let mut stream = ByteStreamReadBuffer::new();
        // Range 0..=5 needs 3 bits, leaving codes 6 and 7 invalid.
        stream.append(&[0b0000_0110]);
        let mut output = VecDeque::new();

        assert!(BitPack::unpack_ints(&mut stream, 0, 5, &mut output, 1).is_err());
        assert!(output.is_empty());
    }

    #[test]
    fn handles_zero_and_full_width_integer_layouts() {
        let mut constant_stream = ByteStreamReadBuffer::new();
        constant_stream.append(&[u8::MAX]);
        let mut constant_output = VecDeque::new();
        BitPack::unpack_ints(&mut constant_stream, 7, 7, &mut constant_output, usize::MAX).unwrap();
        assert!(constant_output.is_empty());
        assert_eq!(constant_stream.available(), 8);

        let mut full_width_stream = ByteStreamReadBuffer::new();
        full_width_stream.append(&u64::MAX.to_le_bytes());
        let mut full_width_output = VecDeque::new();
        BitPack::unpack_ints(
            &mut full_width_stream,
            i64::MIN,
            i64::MAX,
            &mut full_width_output,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(
            full_width_output,
            VecDeque::from([RecordValue::Integer(i64::MAX)])
        );
        assert_eq!(full_width_stream.available(), 0);
    }

    #[test]
    fn record_budget_leaves_final_integer_padding_unconsumed() {
        let mut stream = ByteStreamReadBuffer::new();
        // Two 3-bit records followed by an otherwise-invalid 2-bit suffix.
        stream.append(&[0b1101_0001]);
        let mut output = VecDeque::new();

        BitPack::unpack_ints(&mut stream, -2, 3, &mut output, 2).unwrap();

        assert_eq!(output.len(), 2);
        assert_eq!(stream.available(), 2);
    }

    #[test]
    fn record_budget_applies_to_fixed_width_streams() {
        let mut stream = ByteStreamReadBuffer::new();
        stream.append(&[0_u8; 8]);
        let mut output = VecDeque::new();

        BitPack::unpack_singles(&mut stream, &mut output, 1).unwrap();

        assert_eq!(output, VecDeque::from([RecordValue::Single(0.0)]));
        assert_eq!(stream.available(), 32);
    }
}
