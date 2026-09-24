import sqlite3
import shapely.wkt
from shapely.geometry import shape

def st_intersects(wkt1, wkt2):
    try:
        geom1 = shapely.wkt.loads(wkt1)
        geom2 = shapely.wkt.loads(wkt2)
        return int(geom1.intersects(geom2))
    except Exception:
        return 0

conn = sqlite3.connect(":memory:")
conn.create_function("ST_Intersects", 2, st_intersects)
cursor = conn.cursor()
cursor.execute("CREATE TABLE zones (id INTEGER, geom TEXT)")
cursor.execute("INSERT INTO zones VALUES (1, 'POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')")
cursor.execute("INSERT INTO zones VALUES (2, 'POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20))')")

query_geom = "POINT (5 5)"
cursor.execute("SELECT id FROM zones WHERE ST_Intersects(geom, ?)", (query_geom,))
print("Intersecting IDs:", cursor.fetchall())
