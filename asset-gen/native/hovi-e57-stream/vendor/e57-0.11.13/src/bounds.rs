use crate::xml;
use crate::Result;
use roxmltree::Node;

/// Optional minimum and maximum values for Cartesian X, Y and Z coordinates.
///
/// Represents an axis-aligned bounding box of Cartesian coordinates.
#[derive(Clone, Debug, Default)]
pub struct CartesianBounds {
    pub x_min: Option<f64>,
    pub x_max: Option<f64>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    pub z_min: Option<f64>,
    pub z_max: Option<f64>,
}

impl CartesianBounds {
    pub(crate) fn from_node(node: &Node) -> Result<Self> {
        Ok(Self {
            x_min: xml::opt_f64(node, "xMinimum")?,
            x_max: xml::opt_f64(node, "xMaximum")?,
            y_min: xml::opt_f64(node, "yMinimum")?,
            y_max: xml::opt_f64(node, "yMaximum")?,
            z_min: xml::opt_f64(node, "zMinimum")?,
            z_max: xml::opt_f64(node, "zMaximum")?,
        })
    }

    pub(crate) fn xml_string(&self) -> String {
        let mut xml = String::from("<cartesianBounds type=\"Structure\">\n");
        if let Some(min) = self.x_min {
            xml += &xml::gen_float("xMinimum", min);
        }
        if let Some(max) = self.x_max {
            xml += &xml::gen_float("xMaximum", max);
        }
        if let Some(min) = self.y_min {
            xml += &xml::gen_float("yMinimum", min);
        }
        if let Some(max) = self.y_max {
            xml += &xml::gen_float("yMaximum", max);
        }
        if let Some(min) = self.z_min {
            xml += &xml::gen_float("zMinimum", min);
        }
        if let Some(max) = self.z_max {
            xml += &xml::gen_float("zMaximum", max);
        }
        xml += "</cartesianBounds>\n";
        xml
    }
}

/// Optional minimum and maximum values for spherical coordinates.
#[derive(Clone, Debug, Default)]
pub struct SphericalBounds {
    pub range_min: Option<f64>,
    pub range_max: Option<f64>,
    pub elevation_min: Option<f64>,
    pub elevation_max: Option<f64>,
    pub azimuth_start: Option<f64>,
    pub azimuth_end: Option<f64>,
}

impl SphericalBounds {
    pub(crate) fn from_node(node: &Node) -> Result<Self> {
        Ok(Self {
            range_min: xml::opt_f64(node, "rangeMinimum")?,
            range_max: xml::opt_f64(node, "rangeMaximum")?,
            elevation_min: xml::opt_f64(node, "elevationMinimum")?,
            elevation_max: xml::opt_f64(node, "elevationMaximum")?,
            azimuth_start: xml::opt_f64(node, "azimuthStart")?,
            azimuth_end: xml::opt_f64(node, "azimuthEnd")?,
        })
    }

    pub(crate) fn xml_string(&self) -> String {
        let mut xml = String::from("<sphericalBounds type=\"Structure\">\n");
        if let Some(min) = self.azimuth_start {
            xml += &xml::gen_float("azimuthStart", min);
        }
        if let Some(max) = self.azimuth_end {
            xml += &xml::gen_float("azimuthEnd", max);
        }
        if let Some(min) = self.elevation_min {
            xml += &xml::gen_float("elevationMinimum", min);
        }
        if let Some(max) = self.elevation_max {
            xml += &xml::gen_float("elevationMaximum", max);
        }
        if let Some(min) = self.range_min {
            xml += &xml::gen_float("rangeMinimum", min);
        }
        if let Some(max) = self.range_max {
            xml += &xml::gen_float("rangeMaximum", max);
        }
        xml += "</sphericalBounds>\n";
        xml
    }

    /// Converts the spherical bounds into Cartesian bounds.
    /// The result will be bigger than the actual Cartesian bounds, since it is not possible
    /// to calculate the exact Cartesian bounds without iterating over all points.
    /// Will return `None` if the spherical range is not defined.
    pub fn to_cartesian(&self) -> Option<CartesianBounds> {
        self.range_max.map(|range| CartesianBounds {
            x_min: Some(-range),
            x_max: Some(range),
            y_min: Some(-range),
            y_max: Some(range),
            z_min: Some(-range),
            z_max: Some(range),
        })
    }
}

/// Optional minimum and maximum values for the row, column and return indices.
#[derive(Clone, Debug, Default)]
pub struct IndexBounds {
    pub row_min: Option<i64>,
    pub row_max: Option<i64>,
    pub column_min: Option<i64>,
    pub column_max: Option<i64>,
    pub return_min: Option<i64>,
    pub return_max: Option<i64>,
}

impl IndexBounds {
    pub(crate) fn from_node(node: &Node) -> Result<Self> {
        Ok(Self {
            row_min: xml::opt_int(node, "rowMinimum")?,
            row_max: xml::opt_int(node, "rowMaximum")?,
            column_min: xml::opt_int(node, "columnMinimum")?,
            column_max: xml::opt_int(node, "columnMaximum")?,
            return_min: xml::opt_int(node, "returnMinimum")?,
            return_max: xml::opt_int(node, "returnMaximum")?,
        })
    }

    pub(crate) fn xml_string(&self) -> String {
        let mut xml = String::from("<indexBounds type=\"Structure\">\n");
        if let Some(min) = self.row_min {
            xml += &xml::gen_int("rowMinimum", min);
        }
        if let Some(max) = self.row_max {
            xml += &xml::gen_int("rowMaximum", max);
        }
        if let Some(min) = self.column_min {
            xml += &xml::gen_int("columnMinimum", min);
        }
        if let Some(max) = self.column_max {
            xml += &xml::gen_int("columnMaximum", max);
        }
        if let Some(min) = self.return_min {
            xml += &xml::gen_int("returnMinimum", min);
        }
        if let Some(max) = self.return_max {
            xml += &xml::gen_int("returnMaximum", max);
        }
        xml += "</indexBounds>\n";
        xml
    }
}
