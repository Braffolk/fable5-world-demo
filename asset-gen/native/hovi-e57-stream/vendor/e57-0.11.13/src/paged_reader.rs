use std::io::{Error, ErrorKind, Read, Result, Seek, SeekFrom};

#[cfg(not(feature = "crc32c"))]
use crate::crc32::Crc32;

const CHECKSUM_SIZE: u64 = 4;
const ALIGNMENT_SIZE: u64 = 4;
const MAX_PAGE_SIZE: u64 = 1024 * 1024;

pub struct PagedReader<T: Read + Seek> {
    page_size: u64,
    phy_file_size: u64,
    log_file_size: u64,
    pages: u64,
    reader: T,
    offset: u64,
    page_num: Option<u64>,
    page_buffer: Vec<u8>,

    #[cfg(not(feature = "crc32c"))]
    crc: Crc32,
}

impl<T: Read + Seek> PagedReader<T> {
    /// Create and initialize a paged reader that abstracts the E57 CRC scheme
    pub fn new(mut reader: T, page_size: u64) -> Result<Self> {
        if page_size > MAX_PAGE_SIZE {
            Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Page size {page_size} is bigger than the allowed maximum page size of {MAX_PAGE_SIZE} bytes"),
            ))?;
        }
        if page_size <= CHECKSUM_SIZE {
            let msg = if page_size == 0 {
                String::from("File does not appear to be a valid E57 file (page size is 0, the header may be corrupt or missing)")
            } else {
                format!("Page size {page_size} needs to be bigger than checksum ({CHECKSUM_SIZE} bytes)")
            };
            Err(Error::new(ErrorKind::InvalidInput, msg))?;
        }

        let phy_file_size = reader.seek(SeekFrom::End(0))?;
        if phy_file_size == 0 {
            let msg = "File is empty (0 bytes)";
            Err(Error::new(ErrorKind::InvalidData, msg))?;
        }
        if phy_file_size % page_size != 0 {
            Err(Error::new(
                ErrorKind::InvalidData,
                format!("File appears to be truncated: file size {phy_file_size} is not a multiple of the page size {page_size}"),
            ))?;
        }

        let pages = phy_file_size / page_size;

        Ok(Self {
            reader,
            page_size,
            pages,
            phy_file_size,
            log_file_size: pages * (page_size - CHECKSUM_SIZE),
            page_buffer: vec![0_u8; page_size as usize],
            page_num: None,
            offset: 0,

            #[cfg(not(feature = "crc32c"))]
            crc: Crc32::new(),
        })
    }

    /// Seeking to a physical file address as offset relative to the start of the file.
    /// Will return the new logical offset inside the file or an error.
    pub fn seek_physical(&mut self, offset: u64) -> Result<u64> {
        if offset >= self.phy_file_size {
            Err(Error::new(
                ErrorKind::InvalidInput,
                format!("File appears to be truncated: offset {offset} is behind end of file"),
            ))?;
        }

        let pages_before = offset / self.page_size;
        let page_offset = offset % self.page_size;
        if page_offset >= self.page_size - CHECKSUM_SIZE {
            Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Physical offset {offset} points into a page checksum"),
            ))?;
        }
        self.offset = offset - pages_before * CHECKSUM_SIZE;
        Ok(self.offset)
    }

    /// Converts a physical file offset to a logical payload offset without seeking.
    pub fn logical_offset_for_physical(&self, offset: u64) -> Result<u64> {
        if offset >= self.phy_file_size {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Physical offset {offset} is behind end of file"),
            ));
        }
        let pages_before = offset / self.page_size;
        let page_offset = offset % self.page_size;
        if page_offset >= self.page_size - CHECKSUM_SIZE {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Physical offset {offset} points into a page checksum"),
            ));
        }
        Ok(offset - pages_before * CHECKSUM_SIZE)
    }

    /// Current logical payload position, excluding page checksums.
    pub fn logical_position(&self) -> u64 {
        self.offset
    }

    /// Logical payload size, excluding page checksums.
    pub fn logical_size(&self) -> u64 {
        self.log_file_size
    }

    fn read_page(&mut self, page: u64) -> Result<()> {
        if page >= self.pages {
            let max = self.pages - 1;
            Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Page {page} does not exist, only page numbers 0..{max} are valid"),
            ))?;
        }
        let offset = page * self.page_size;
        self.reader.seek(SeekFrom::Start(offset))?;
        self.reader.read_exact(&mut self.page_buffer)?;
        let data_size = self.page_size - CHECKSUM_SIZE;
        let expected_checksum = &self.page_buffer[data_size as usize..];

        // Simple & slower default included SW implementation
        #[cfg(not(feature = "crc32c"))]
        let crc = self.crc.calculate(&self.page_buffer[0..data_size as usize]);

        // Optional faster external crate with HW support
        #[cfg(feature = "crc32c")]
        let crc = crc32c::crc32c(&self.page_buffer[0..data_size as usize]);

        // The standard says all binary values are stored as little endian,
        // but for some reason E57 files contain the checksum in big endian order.
        // Probably the reference implementation used a weird CRC library and
        // now everybody has to swap bytes as well because it was not noticed back then :)
        let calculated_checksum = crc.to_be_bytes();

        if expected_checksum != calculated_checksum {
            self.page_num = None;
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Detected invalid checksum (expected: {expected_checksum:?}, actual: {calculated_checksum:?}) for page {page}")
            ));
        }

        self.page_num = Some(page);
        Ok(())
    }

    /// Do some skipping to next 4-byte-aligned offset, if needed.
    pub fn align(&mut self) -> Result<()> {
        let off_alignment = self.offset % 4;
        if off_alignment != 0 {
            let skip = ALIGNMENT_SIZE - off_alignment;
            if self.offset + skip > self.log_file_size {
                Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Tried to seek behind end of the file",
                ))?
            }
            self.offset += skip;
        }
        Ok(())
    }
}

impl<T: Read + Seek> Read for PagedReader<T> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let page = self.offset / (self.page_size - CHECKSUM_SIZE);
        if page >= self.pages {
            return Ok(0);
        }
        if self.page_num != Some(page) {
            self.read_page(page)?;
        }
        let page_offset = self.offset % (self.page_size - CHECKSUM_SIZE);
        let page_readable = self.page_size - CHECKSUM_SIZE - page_offset;
        let read_size = usize::min(buf.len(), page_readable as usize);
        buf[..read_size].copy_from_slice(
            &self.page_buffer[page_offset as usize..page_offset as usize + read_size],
        );
        self.offset += read_size as u64;
        Ok(read_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Cursor;

    const PAGE_SIZE: u64 = 1024;

    #[test]
    fn read_full_valid_file() {
        let file_size = 743424_u64;
        let pages = file_size / PAGE_SIZE;
        let logical_file_size = file_size - pages * CHECKSUM_SIZE;
        let file = File::open("testdata/bunnyDouble.e57").unwrap();
        let mut reader = PagedReader::new(file, PAGE_SIZE).unwrap();

        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        assert_eq!(buf.len(), logical_file_size as usize);
    }

    #[test]
    fn size_not_multiple_of_page() {
        let file = File::open("testdata/bunnyDouble.e57").unwrap();
        assert!(PagedReader::new(file, PAGE_SIZE - 1).is_err());
    }

    #[test]
    fn page_size_too_small() {
        let file = File::open("testdata/bunnyDouble.e57").unwrap();
        assert!(PagedReader::new(file, CHECKSUM_SIZE).is_err());
    }

    #[test]
    fn zero_pages() {
        let file = Vec::<u8>::new();
        let cursor = Cursor::new(file);
        assert!(PagedReader::new(cursor, PAGE_SIZE).is_err());
    }

    #[test]
    fn corrupt_page() {
        let data = vec![0_u8; 128];
        let cursor = Cursor::new(data);
        let mut reader = PagedReader::new(cursor, 128).unwrap();

        let mut buf = Vec::new();
        assert!(reader.read_to_end(&mut buf).is_err());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn physical_seek() {
        let file = File::open("testdata/bunnyDouble.e57").unwrap();
        let mut reader = PagedReader::new(file, PAGE_SIZE).unwrap();

        let xml_physical_offset = 740736;
        let expected_logical_offset = 737844;

        let logical_offset = reader.seek_physical(xml_physical_offset).unwrap();
        assert_eq!(logical_offset, expected_logical_offset);

        let mut buffer = [0_u8; 5];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer.to_vec()).unwrap(), "<?xml");
    }

    #[test]
    fn read_end() {
        let file = File::open("testdata/bunnyDouble.e57").unwrap();
        let mut reader = PagedReader::new(file, PAGE_SIZE).unwrap();

        // Read entire file
        let mut buffer = Vec::new();
        assert_eq!(reader.read_to_end(&mut buffer).unwrap(), 740520);

        // Try again, nothing should be read
        buffer.clear();
        assert_eq!(reader.read_to_end(&mut buffer).unwrap(), 0);
    }

    #[test]
    fn align() {
        let data = vec![0_u8; 128];
        let cursor = Cursor::new(data);
        let mut reader = PagedReader::new(cursor, 128).unwrap();

        reader.align().unwrap();
        assert_eq!(reader.offset, 0);

        reader.seek_physical(1).unwrap();
        reader.align().unwrap();
        assert_eq!(reader.offset, ALIGNMENT_SIZE);

        reader.align().unwrap();
        assert_eq!(reader.offset, ALIGNMENT_SIZE);
    }
}
