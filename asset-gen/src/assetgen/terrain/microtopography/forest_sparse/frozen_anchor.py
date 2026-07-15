"""Single deliberate preregistration trust root for the forest sparse screen.

Updating this file is an explicit scientific-method change: refresh every mapped
file hash, then record this file's own hash in the full config and specification.
The module cannot and does not self-hash; the accepting repository commit seals it.
"""

EXPECTED_SEMANTIC_SHA256 = "979fbbd24839b0cc4285042515c0dec809a75af34eaa4a94fc3884846d103f2b"

EXPECTED_EXTERNAL_MODULE_ORIGINS = {
    "assetgen.process.microtopo.model": "asset-gen/src/assetgen/process/microtopo/model.py",
}

# Project-relative paths for every other outcome-producing package file.
EXPECTED_IMPLEMENTATION_SHA256 = {
    "asset-gen/src/assetgen/terrain/microtopography/forest_sparse/__init__.py": "e286b8805be4e8665ec28f079c913cb3ec8d382d3d3ffbeb399d01aa9fde0017",
    "asset-gen/src/assetgen/terrain/microtopography/forest_sparse/__main__.py": "4e58b4ef5ddc25626b3e4ad5c8e9e84e7261900a819ccd65e4b4da953f7dd4eb",
    "asset-gen/src/assetgen/terrain/microtopography/forest_sparse/contracts.py": "22963db58da5047af82fb864f69ae8d889158eda5d171a026e102c52f9dc5097",
    "asset-gen/src/assetgen/terrain/microtopography/forest_sparse/operators.py": "4e47c04fe5e3a38934101ad207c6fb5b8e6bf562b432cb63d7e0145c0dbba53e",
    "asset-gen/src/assetgen/terrain/microtopography/forest_sparse/qa.py": "25407879f573e34cab7ec1ebdf251fe65cfbbbe739c516e7c7ec922215dee7f6",
    "asset-gen/src/assetgen/terrain/microtopography/forest_sparse/screen.py": "f1911315c1b41562d0a2f359e38cb2ee68f138f92735e45d1a4382792912278e",
}
