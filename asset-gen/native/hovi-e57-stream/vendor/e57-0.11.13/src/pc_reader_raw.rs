use crate::paged_reader::PagedReader;
use crate::queue_reader::QueueReader;
use crate::PointCloud;
use crate::RawValues;
use crate::Result;
use std::io::{Read, Seek};

/// Iterate over all raw points of a point cloud for reading.
pub struct PointCloudReaderRaw<'a, T: Read + Seek> {
    queue_reader: QueueReader<'a, T>,
    prototype_len: usize,
    records: u64,
    read: u64,
    failed: bool,
}

impl<'a, T: Read + Seek> PointCloudReaderRaw<'a, T> {
    pub(crate) fn new(pc: &PointCloud, reader: &'a mut PagedReader<T>) -> Result<Self> {
        let queue_reader = QueueReader::new(pc, reader)?;
        let prototype_len = pc.prototype.len();
        let records = pc.records;
        Ok(Self {
            queue_reader,
            prototype_len,
            records,
            read: 0,
            failed: false,
        })
    }

    /// Reads the next raw point into a caller-owned buffer.
    ///
    /// This is equivalent to [`Iterator::next`] without allocating a new
    /// [`RawValues`] vector for every point. The buffer is cleared before a
    /// point is written and retains its allocation for the next call.
    pub fn next_into(&mut self, point: &mut RawValues) -> Option<Result<()>> {
        if self.failed || self.read >= self.records {
            return None;
        }

        while self.queue_reader.available() < 1 {
            if let Err(err) = self.queue_reader.advance() {
                self.failed = true;
                return Some(Err(err));
            }
        }

        match self.queue_reader.pop_point(point) {
            Ok(()) => {
                self.read += 1;
                Some(Ok(()))
            }
            Err(err) => {
                self.failed = true;
                Some(Err(err))
            }
        }
    }
}

impl<T: Read + Seek> Iterator for PointCloudReaderRaw<'_, T> {
    /// Each iterator item is a result for an extracted point.
    type Item = Result<RawValues>;

    /// Returns the next available point or None if the end was reached.
    fn next(&mut self) -> Option<Self::Item> {
        let mut point = RawValues::with_capacity(self.prototype_len);
        match self.next_into(&mut point)? {
            Ok(()) => Some(Ok(point)),
            Err(err) => Some(Err(err)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let overall = self.records;
        let remaining = if self.failed { 0 } else { overall - self.read };
        (remaining as usize, Some(remaining as usize))
    }
}
