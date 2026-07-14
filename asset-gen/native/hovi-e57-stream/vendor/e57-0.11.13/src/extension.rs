use crate::{Error, Record, RecordName, Result};
use roxmltree::Document;

/// Describes an E57 extension by name and URL.
///
/// The E57 specification includes an mechanism for extensions.
/// Each extension has its own namespace in the XML section of the E57 file.
/// Such extensions can for example specify custom point attributes or
/// add additional metadata and custom binary blobs.
///
/// Every E57 parser must be able to ignore any unknown extensions.
/// Some extensions are <a href="http://www.libe57.org/extensions.html" target="_blank">officially documented</a>,
/// others are proprietary and have no public documentation.
///
/// Since full extension support involves all kinds of XML operations,
/// it can greatly increase the API of any E57 library.
/// This library is using a more pragmatic approach and requires you to bring your own XML library.
/// This allows the API of this library to stay small, focused and lightweight.
///
/// Extension features directly supported by this library are:
/// * reading and defining of XML namespaces for extensions
/// * reading and writing additional custom point attributes
/// * reading and writing of binary blobs
///
/// Extensions that require specific XML parsing are possible.
/// You need to load your E57 file and then call `E57Reader::xml()` method to get the full original XML string.
/// This will return an UTF8 string that can be feed into an XML parser.
/// This library is using `roxmltree` for lightweight XML parsing.
///
/// Extensions that require XML manipulation when writing E57 files are also possible.
/// You need to first finishing writing all point clouds, images and binary blobs.
/// Then when you are ready to call `E57Writer::finalize()` to write the XML section and close the file,
/// you need to call `E57Writer::finalize_customized_xml()` instead.
/// This allows you to supply a transformer that will receive the generated XML string
/// and can manipulate it before its written into the file.
/// Your code is responsible for parsing, modifying and serializing th XML again in a non-destructive way!
///
/// # Example Code
/// You can find a <a href="https://github.com/cry-inc/e57/blob/master/tests/extensions.rs" target="_blank">
/// complete example</a> for reading and writing E57 files with extensions in the automated tests of the library.
#[derive(Clone, Debug)]
pub struct Extension {
    /// XML namespace name.
    pub namespace: String,
    /// XML namespace URL.
    pub url: String,
}

impl Extension {
    /// Intialize and return a new Extension structure with the given values.
    pub fn new(namespace: &str, url: &str) -> Self {
        Self {
            namespace: namespace.to_owned(),
            url: url.to_owned(),
        }
    }

    pub(crate) fn vec_from_document(document: &Document) -> Vec<Extension> {
        let mut extensions = Vec::new();
        for item in document.root_element().namespaces() {
            if let Some(name) = item.name() {
                extensions.push(Extension {
                    namespace: name.to_string(),
                    url: item.uri().to_string(),
                });
            }
        }
        extensions
    }

    pub(crate) fn validate_prototype(prototype: &[Record], extensions: &[Extension]) -> Result<()> {
        for record in prototype {
            if let RecordName::Unknown { namespace, name } = &record.name {
                Self::validate_name(namespace)?;
                Self::validate_name(name)?;
                if !extensions.iter().any(|e| &e.namespace == namespace) {
                    Error::invalid(format!(
                        "Cannot find extension namespace {namespace} used by attribute {name}, please register extension first"
                    ))?
                }
            }
        }
        Ok(())
    }

    pub(crate) fn validate_name(name: &str) -> Result<()> {
        if name.is_empty() {
            Error::invalid("Strings used as XML namespaces or attributes must not be empty")?
        }
        if name.to_lowercase().starts_with("xml") {
            Error::invalid(format!(
                "Strings used as XML namespaces or attributes must not start with 'XML': {name}"
            ))?
        }
        let valid_chars = name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || (c == '_') || (c == '-'));
        if !valid_chars {
            Error::invalid(
                format!("Strings used as XML namespaces or attributes should consist only of a-z, A-Z, 0-9, dashes and underscores: '{name}'"),
            )?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_name() {
        assert!(Extension::validate_name("abcz").is_ok());
        assert!(Extension::validate_name("ABCZ").is_ok());
        assert!(Extension::validate_name("0129").is_ok());
        assert!(Extension::validate_name("-_-").is_ok());
        assert!(Extension::validate_name("aBC-DEf-Z_09").is_ok());

        assert!(Extension::validate_name("xmlabc").is_err());
        assert!(Extension::validate_name("XMLabc").is_err());
        assert!(Extension::validate_name("axml").is_ok());

        assert!(Extension::validate_name("abc.").is_err());
        assert!(Extension::validate_name("äüöß").is_err());

        assert!(Extension::validate_name("").is_err());
    }
}
