use crate::error::Converter;
use crate::{DateTime, Error, Result, Transform};
use roxmltree::Node;
use std::fmt::Display;
use std::str::FromStr;

pub fn opt_string(parent_node: &Node, tag_name: &str) -> Result<Option<String>> {
    if let Some(tag) = parent_node.children().find(|n| n.has_tag_name(tag_name)) {
        let expected_type = "String";
        if let Some(found_type) = tag.attribute("type") {
            if found_type != expected_type {
                Error::invalid(format!(
                    "Found XML tag '{tag_name}' with type '{found_type}' instead of '{expected_type}'"
                ))?
            }
        } else {
            Error::invalid(format!("XML tag '{tag_name}' has no 'type' attribute"))?
        }
        let text = tag.text().unwrap_or("");
        Ok(Some(text.to_string()))
    } else {
        Ok(None)
    }
}

pub fn req_string(parent_node: &Node, tag_name: &str) -> Result<String> {
    let str = opt_string(parent_node, tag_name)?;
    str.invalid_err(format!("XML tag '{tag_name}' was not found"))
}

fn opt_num<T: FromStr + Sync + Send>(
    parent_node: &Node,
    tag_name: &str,
    expected_type: &str,
) -> Result<Option<T>> {
    if let Some(tag) = parent_node.children().find(|n| n.has_tag_name(tag_name)) {
        if let Some(found_type) = tag.attribute("type") {
            if found_type != expected_type {
                Error::invalid(format!(
                    "Found XML tag '{tag_name}' with type '{found_type}' instead of '{expected_type}'"
                ))?
            }
        } else {
            Error::invalid(format!("XML tag '{tag_name}' has no 'type' attribute"))?
        }
        let text = tag.text().unwrap_or("0");
        if let Ok(parsed) = text.parse::<T>() {
            Ok(Some(parsed))
        } else {
            Error::invalid(format!(
                "Cannot parse value '{text}' of XML tag '{tag_name}' as '{expected_type}'"
            ))?
        }
    } else {
        Ok(None)
    }
}

pub fn opt_f64(parent_node: &Node, tag_name: &str) -> Result<Option<f64>> {
    opt_num(parent_node, tag_name, "Float")
}

pub fn req_f64(parent_node: &Node, tag_name: &str) -> Result<f64> {
    let double = opt_num(parent_node, tag_name, "Float")?;
    double.invalid_err(format!("XML tag '{tag_name}' was not found"))
}

pub fn opt_int<T: FromStr + Sync + Send>(parent_node: &Node, tag_name: &str) -> Result<Option<T>> {
    opt_num(parent_node, tag_name, "Integer")
}

pub fn req_int<T: FromStr + Send + Sync>(parent_node: &Node, tag_name: &str) -> Result<T> {
    let integer = opt_num(parent_node, tag_name, "Integer")?;
    integer.invalid_err(format!("XML tag '{tag_name}' was not found"))
}

pub fn opt_date_time(parent_node: &Node, tag_name: &str) -> Result<Option<DateTime>> {
    if let Some(tag) = parent_node.children().find(|n| n.has_tag_name(tag_name)) {
        let expected_type = "Structure";
        if let Some(found_type) = tag.attribute("type") {
            if found_type != expected_type {
                Error::invalid(format!(
                    "Found XML tag '{tag_name}' with type '{found_type}' instead of '{expected_type}'"
                ))?
            }
        } else {
            Error::invalid(format!("XML tag '{tag_name}' has no 'type' attribute"))?
        }
        DateTime::from_node(&tag)
    } else {
        Ok(None)
    }
}

pub fn opt_transform(parent_node: &Node, tag_name: &str) -> Result<Option<Transform>> {
    let node = parent_node.children().find(|n| n.has_tag_name(tag_name));
    if let Some(node) = node {
        Ok(Some(Transform::from_node(&node)?))
    } else {
        Ok(None)
    }
}

pub fn gen_string<T: Display>(tag_name: &str, value: &T) -> String {
    format!("<{tag_name} type=\"String\"><![CDATA[{value}]]></{tag_name}>\n")
}

pub fn gen_float<T: Display>(tag_name: &str, value: T) -> String {
    format!("<{tag_name} type=\"Float\">{value}</{tag_name}>\n")
}

pub fn gen_int<T: Display>(tag_name: &str, value: T) -> String {
    format!("<{tag_name} type=\"Integer\">{value}</{tag_name}>\n")
}
