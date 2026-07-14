use crate::args::InputSpec;
use crate::error::{AppError, AppResult};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;

const HEADER_BYTES: usize = 48;

pub struct OpenInput {
    pub file: File,
    pub mode: &'static str,
    pub bytes: u64,
    pub device: u64,
    pub inode: u64,
    pub header: HeaderPreflight,
}

#[derive(Clone, Copy)]
pub struct HeaderPreflight {
    pub major: u32,
    pub minor: u32,
    pub physical_length: u64,
    pub xml_offset: u64,
    pub xml_length: u64,
    pub page_size: u64,
}

pub fn open_and_preflight(
    input: &InputSpec,
    max_file_bytes: u64,
    max_xml_bytes: u64,
) -> AppResult<OpenInput> {
    let (mut file, mode) = match input {
        InputSpec::Path(path) => (
            File::open(path).map_err(|error| {
                AppError::input(format!("failed to open input {}: {error}", path.display()))
            })?,
            "path",
        ),
        InputSpec::FileDescriptor(descriptor) => {
            let path = format!("/dev/fd/{descriptor}");
            (
                File::open(&path).map_err(|error| {
                    AppError::input(format!("failed to duplicate inherited {path}: {error}"))
                })?,
                "inherited_fd",
            )
        }
    };
    let metadata = file
        .metadata()
        .map_err(|error| AppError::input(format!("failed to stat input: {error}")))?;
    if !metadata.is_file() {
        return Err(AppError::input("input descriptor is not a regular file"));
    }
    let bytes = metadata.len();
    if bytes == 0 || bytes > max_file_bytes {
        return Err(AppError::validation(format!(
            "input size {bytes} is outside the configured 1..={max_file_bytes} byte range"
        )));
    }

    let mut encoded = [0_u8; HEADER_BYTES];
    file.read_exact(&mut encoded)
        .map_err(|error| AppError::input(format!("failed to read E57 header: {error}")))?;
    let header = parse_header(&encoded)?;
    if header.physical_length != bytes {
        return Err(AppError::validation(format!(
            "E57 header physical length {} differs from descriptor length {bytes}",
            header.physical_length
        )));
    }
    if header.xml_length == 0 || header.xml_length > max_xml_bytes {
        return Err(AppError::validation(format!(
            "E57 XML length {} is outside the configured 1..={max_xml_bytes} byte range",
            header.xml_length
        )));
    }
    if header.xml_offset >= bytes || header.xml_length > bytes {
        return Err(AppError::validation(
            "E57 XML offset or logical length is outside the input file",
        ));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| AppError::input(format!("failed to rewind E57 input: {error}")))?;
    Ok(OpenInput {
        file,
        mode,
        bytes,
        device: metadata.dev(),
        inode: metadata.ino(),
        header,
    })
}

fn parse_header(encoded: &[u8; HEADER_BYTES]) -> AppResult<HeaderPreflight> {
    if &encoded[0..8] != b"ASTM-E57" {
        return Err(AppError::validation("input has no ASTM-E57 signature"));
    }
    let major = u32::from_le_bytes(array(encoded, 8)?);
    let minor = u32::from_le_bytes(array(encoded, 12)?);
    let physical_length = u64::from_le_bytes(array(encoded, 16)?);
    let xml_offset = u64::from_le_bytes(array(encoded, 24)?);
    let xml_length = u64::from_le_bytes(array(encoded, 32)?);
    let page_size = u64::from_le_bytes(array(encoded, 40)?);
    if major != 1 || minor != 0 || page_size != 1024 {
        return Err(AppError::validation(format!(
            "unsupported E57 header version {major}.{minor} or page size {page_size}"
        )));
    }
    if physical_length == 0 || !physical_length.is_multiple_of(page_size) {
        return Err(AppError::validation(
            "E57 physical length is zero or not page aligned",
        ));
    }
    Ok(HeaderPreflight {
        major,
        minor,
        physical_length,
        xml_offset,
        xml_length,
        page_size,
    })
}

fn array<const N: usize>(encoded: &[u8], offset: usize) -> AppResult<[u8; N]> {
    encoded
        .get(offset..offset + N)
        .ok_or_else(|| AppError::validation("E57 header field is truncated"))?
        .try_into()
        .map_err(|_| AppError::validation("E57 header field has the wrong width"))
}
