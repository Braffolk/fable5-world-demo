use std::convert::Infallible;
use std::error::Error as StdError;
use std::fmt::Result as FmtResult;
use std::fmt::{Display, Formatter};
use std::result::Result as StdResult;

/// To be used as error message when extracting stuff from arrays that should never fail
pub const WRONG_OFFSET: &str = "Wrong buffer offset detected";

/// Possible errors that can occur while working with E57 files.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The E57 you are reading or creating is invalid and does not confirm with the E57 format specification.
    Invalid {
        desc: String,
        source: Option<Box<dyn StdError + Send + Sync + 'static>>,
    },

    /// Something went wrong while reading data from an E57 file.
    /// Typically this is caused by an IO error outside the library or because of an incomplete file.
    Read {
        desc: String,
        source: Option<Box<dyn StdError + Send + Sync + 'static>>,
    },

    /// Something went wrong while writing data to an E57 file.
    /// Typically this is caused by an IO error outside the library.
    Write {
        desc: String,
        source: Option<Box<dyn StdError + Send + Sync + 'static>>,
    },

    /// Some feature or aspect of E57 that is not yet implement by this library.
    /// Feel free to create feature request issues here: <https://github.com/cry-inc/e57/issues>
    NotImplemented { desc: String },

    /// An unexpected internal issue occured.
    /// Most likely this is a logic bug inside the library.
    /// Please file an issue on GitHub, if possible.
    Internal {
        desc: String,
        source: Option<Box<dyn StdError + Send + Sync + 'static>>,
    },
}

impl Error {
    /// Creates a new Invalid error.
    pub fn invalid<T, C>(desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        Err(Error::Invalid {
            desc: desc.to_string(),
            source: None,
        })
    }

    /// Creates a new `NotImplemented` error.
    pub fn not_implemented<T, C>(desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        Err(Error::NotImplemented {
            desc: desc.to_string(),
        })
    }

    /// Creates a new Internal error.
    pub fn internal<T, C>(desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        Err(Error::Internal {
            desc: desc.to_string(),
            source: None,
        })
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Error::Invalid { desc, source } => match source {
                Some(src) => write!(f, "Invalid E57 content: {desc}: {src}"),
                None => write!(f, "Invalid E57 content: {desc}"),
            },
            Error::Read { desc, source } => match source {
                Some(src) => write!(f, "Failed to read E57: {desc}: {src}"),
                None => write!(f, "Failed to read E57: {desc}"),
            },
            Error::Internal { desc, source } => match source {
                Some(src) => write!(f, "Internal error: {desc}: {src}"),
                None => write!(f, "Internal error: {desc}"),
            },
            Error::NotImplemented { desc } => write!(f, "Not implemented: {desc}"),
            Error::Write { desc, source } => match source {
                Some(src) => write!(f, "Failed to write E57: {desc}: {src}"),
                None => write!(f, "Failed to write E57: {desc}"),
            },
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Invalid { source, .. } => source
                .as_ref()
                .map(|s| s.as_ref() as &(dyn StdError + 'static)),
            Error::Read { source, .. } => source
                .as_ref()
                .map(|s| s.as_ref() as &(dyn StdError + 'static)),
            Error::Internal { source, .. } => source
                .as_ref()
                .map(|s| s.as_ref() as &(dyn StdError + 'static)),
            Error::Write { source, .. } => source
                .as_ref()
                .map(|s| s.as_ref() as &(dyn StdError + 'static)),
            Error::NotImplemented { .. } => None,
        }
    }
}

/// Custom result type hardwired to use the Error type of this crate.
pub type Result<T> = StdResult<T, Error>;

/// Helper trait for types that can be converted into an Error.
pub trait Converter<T, E> {
    fn read_err<C>(self, context: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static;

    fn invalid_err<C>(self, context: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static;

    fn internal_err<C>(self, context: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static;

    fn write_err<C>(self, context: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static;
}

/// Create an library Error from std Error instances.
impl<T, E> Converter<T, E> for StdResult<T, E>
where
    E: StdError + Send + Sync + 'static,
{
    fn read_err<C>(self, desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::Read {
                desc: desc.to_string(),
                source: Some(Box::new(error)),
            }),
        }
    }

    fn invalid_err<C>(self, desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::Invalid {
                desc: desc.to_string(),
                source: Some(Box::new(error)),
            }),
        }
    }

    fn internal_err<C>(self, desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::Internal {
                desc: desc.to_string(),
                source: Some(Box::new(error)),
            }),
        }
    }

    fn write_err<C>(self, desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::Write {
                desc: desc.to_string(),
                source: Some(Box::new(error)),
            }),
        }
    }
}

/// Create an library Error from Option instances.
impl<T> Converter<T, Infallible> for Option<T> {
    fn read_err<C>(self, desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::Read {
                desc: desc.to_string(),
                source: None,
            }),
        }
    }

    fn invalid_err<C>(self, desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::Invalid {
                desc: desc.to_string(),
                source: None,
            }),
        }
    }

    fn internal_err<C>(self, desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::Internal {
                desc: desc.to_string(),
                source: None,
            }),
        }
    }

    fn write_err<C>(self, desc: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::Write {
                desc: desc.to_string(),
                source: None,
            }),
        }
    }
}
