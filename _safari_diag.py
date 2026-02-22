import sqlite3, shutil, tempfile, os
from pathlib import Path

db = Path.home() / "Library" / "Safari" / "History.db"
tmp = tempfile.mktemp(suffix=".db")
shutil.copy2(db, tmp)
conn = sqlite3.connect(tmp)
conn.row_factory = sqlite3.Row

for ext in ('.pdf', '.docx', '.xlsx', '.html'):
    n = conn.execute(f"SELECT COUNT(*) FROM history_items WHERE url LIKE '%{ext}%'").fetchone()[0]
    print(f"history_items with {ext}: {n}")

print("\nSample PDF visits with redirect info:")
rows = conn.execute("""
    SELECT hi.url, hv.title, hv.redirect_source, hv.redirect_destination, hv.visit_time
    FROM history_visits hv
    JOIN history_items hi ON hi.id = hv.history_item
    WHERE hi.url LIKE '%.pdf%'
    LIMIT 10
""").fetchall()
for r in rows:
    src = r['redirect_source']
    src_url = None
    if src:
        row2 = conn.execute(
            "SELECT hi2.url FROM history_visits hv2 JOIN history_items hi2 ON hi2.id = hv2.history_item WHERE hv2.id = ?",
            (src,)
        ).fetchone()
        src_url = row2['url'] if row2 else None
    print(f"  url={r['url'][:90]}")
    print(f"    title={r['title']}  redirect_src_id={src}  src_url={src_url}  redirect_dst={r['redirect_destination']}")

print("\nAttributes sample:")
for s in conn.execute("SELECT attributes, http_non_get FROM history_visits LIMIT 5").fetchall():
    print(f"  attributes={s['attributes']}  http_non_get={s['http_non_get']}")

print("\nSize-like columns:")
for t in ['history_items', 'history_visits']:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})").fetchall()]
    size_cols = [c for c in cols if any(k in c.lower() for k in ('size', 'byte', 'length', 'content'))]
    print(f"  {t}: {size_cols or '(none)'}")

conn.close()
os.unlink(tmp)
