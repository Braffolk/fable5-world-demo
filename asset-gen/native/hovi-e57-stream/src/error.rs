use std::fmt::{Display, Formatter};

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug)]
pub struct AppError {
    pub code: i32,
    pub kind: &'static str,
    pub message: String,
}

impl AppError {
    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(2, "usage", message)
    }

    pub fn input(message: impl Into<String>) -> Self {
        Self::new(3, "input", message)
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self::new(4, "validation", message)
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(5, "unavailable", message)
    }

    pub fn output(message: impl Into<String>) -> Self {
        Self::new(6, "output", message)
    }

    fn new(code: i32, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            kind,
            message: message.into(),
        }
    }
}

impl Display for AppError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for AppError {}
