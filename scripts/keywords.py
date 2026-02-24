\
"""

Liste de mots-clés pour classifier les PDFs.
Le principe : on cherche des occurrences dans le texte extrait.

"""

REGION_KEYWORDS = {
    "tibial_plateau": [
        "tibial plateau", "plateau tibial", "plateaux tibiaux",
        "proximal tibia", "tibia plateau", "schatzker", "tibial condyle"
    ],
    "pelvis": [
        "pelvis", "pelvic", "acetabulum", "acetabular",
        "pubic rami", "iliac", "ischium", "sacroiliac", "sacrum"
    ],
    "radius": [
        "radius", "radial", "distal radius", "proximal radius",
        "colles", "smith", "barton", "radial head", "radial styloid"
    ],
}


FRACTURE_TYPE_KEYWORDS = {
    "ao_ota": ["ao/ota", "ao ota", "ota", "arbeitsgemeinschaft", "orthopaedic trauma association"],
    "schatzker": ["schatzker"],
    "intra_articular": ["intra-articular", "intra articular"],
    "extra_articular": ["extra-articular", "extra articular"],
    "open_fracture": ["open fracture", "gustilo", "compound fracture"],
    "comminuted": ["comminuted", "multi-fragment", "multifragment"],
    "displaced": ["displaced", "dislocation", "subluxation"],
}

 
LOCATION_KEYWORDS = {
    "medial": ["medial"],
    "lateral": ["lateral"],
    "bicondylar": ["bicondylar", "bicolumn", "both columns"],
    "posterior": ["posterior"],
    "anterior": ["anterior"],
    "acetabulum": ["acetabulum", "acetabular"],
    "pubic_rami": ["pubic rami", "pubic ramus"],
    "radial_head": ["radial head"],
    "radial_styloid": ["radial styloid"],
    "distal": ["distal"],
    "proximal": ["proximal"],
}
