# Sample graph sources

The `.mtx` files are unchanged copies of the datasets already present under the research repository's `data/` directory. Their original Matrix Market comments are retained.

- **USpowerGrid**: [SuiteSparse / Pajek / USpowerGrid](https://sparse.tamu.edu/Pajek/USpowerGrid), P. Tsaparas; Pajek adaptation by V. Batagelj (2006). 4,941 vertices, 6,594 undirected edges (13,188 symmetric matrix entries). Original sources and attribution are included in the MTX header.
- **luxembourg_osm**: [SuiteSparse / DIMACS10 / luxembourg_osm](https://sparse.tamu.edu/DIMACS10/luxembourg_osm), Geofabrik GmbH; editor M. Kobitzsh (2010), DIMACS10 street-network dataset. 114,599 vertices, 119,666 undirected edges (239,332 symmetric matrix entries). © OpenStreetMap contributors. OpenStreetMap data is available under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/); see [OpenStreetMap copyright and attribution](https://www.openstreetmap.org/copyright).

No graph vertices are sampled or truncated for the browser samples. The browser applies the same undirected, unweighted largest-connected-component preprocessing to samples and uploaded files.
