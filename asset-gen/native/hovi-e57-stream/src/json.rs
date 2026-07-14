use crate::error::{AppError, AppResult};

pub fn push_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            value if value <= '\u{1f}' => {
                output.push_str(&format!("\\u{:04x}", value as u32));
            }
            value => output.push(value),
        }
    }
    output.push('"');
}

pub fn error_json(error: &AppError) -> String {
    let mut output = String::from("{\"schemaVersion\":\"laas-hovi-e57-error/1.0.0\",\"kind\":");
    push_string(&mut output, error.kind);
    output.push_str(",\"message\":");
    push_string(&mut output, &error.message);
    output.push_str("}\n");
    output
}

pub fn ensure_limit(output: &str, limit: usize) -> AppResult<()> {
    if output.len() > limit {
        return Err(AppError::output(format!(
            "metadata JSON exceeded the {limit}-byte ceiling"
        )));
    }
    Ok(())
}
