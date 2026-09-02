"""Auditoria de REDES: de cuantas redes distintas se conecta la gente al dashboard.

Consulta interna. Lee el registro que la app escribe en redes.json, dentro del mismo Gist
secreto que el contador de visitas, y lo saca por pantalla. La app no lo pinta en ningun
sitio: el numero de la barra lateral es el de visitantes y este fichero no lo toca.

Las redes van seudonimizadas: la IP nunca se guarda en claro, solo un HMAC-SHA256 truncado a
12 hex. Eso significa que aqui NO se puede saber que IP es cada una, a proposito. Lo que si se
puede saber, que es lo que se buscaba, es CUANTAS son y cada cuanto vuelve cada una.

  python audit_scripts/audit_redes.py            tabla por pantalla
  python audit_scripts/audit_redes.py --json     el registro crudo
"""
import json, re, sys, urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SECRETOS = RAIZ / "streamlit" / ".streamlit" / "secrets.toml"

if not SECRETOS.exists():
    sys.exit(f"No hay secretos en {SECRETOS}. Ver README, «Contador de visitas».")
txt = SECRETOS.read_text(encoding="utf-8")

def secreto(clave):
    m = re.search(rf'^\s*{clave}\s*=\s*"([^"]+)"', txt, re.M)
    return m.group(1) if m else ""

gid, token = secreto("gist_visitas_id"), secreto("gist_visitas_token")
if not (gid and token):
    sys.exit("Faltan gist_visitas_id o gist_visitas_token en secrets.toml.")

req = urllib.request.Request(
    f"https://api.github.com/gists/{gid}",
    headers={"Authorization": f"Bearer {token}",
             "Accept": "application/vnd.github+json",
             "User-Agent": "audit-redes"})
with urllib.request.urlopen(req, timeout=20) as r:
    ficheros = json.loads(r.read())["files"]

visitas = json.loads(ficheros["visitas.json"]["content"])["visitas"]
crudo = ficheros.get("redes.json")
if not crudo:
    sys.exit("Todavia no hay redes.json en el Gist: la app no ha recibido ninguna visita\n"
             "de un navegador real desde que se desplego el registro.")
reg = json.loads(crudo["content"])

if "--json" in sys.argv:
    print(json.dumps(reg, indent=1, sort_keys=True))
    raise SystemExit

redes = reg.get("redes") or {}
print(f"Actualizado    : {reg.get('actualizado', '?')}")
print(f"Visitantes     : {visitas}   (el numero que sale en la barra lateral)")
print(f"Redes distintas: {len(redes)}")
if not redes:
    raise SystemExit

sesiones = sum(int(v.get("n") or 0) for v in redes.values())
print(f"Sesiones vistas: {sesiones}   (visitas de navegador real, incluidas las repetidas)")
print(f"\n{'red (seudonimo)':<16} {'sesiones':>8}  {'primera vez':<21} {'ultima vez':<21}")
print("-" * 70)
for h, v in sorted(redes.items(), key=lambda kv: int(kv[1].get("n") or 0), reverse=True):
    print(f"{h:<16} {int(v.get('n') or 0):>8}  "
          f"{str(v.get('primera', '?')):<21} {str(v.get('ultima', '?')):<21}")

# Una red que vuelve muchas veces suele ser la propia: conviene verlo separado del resto.
if len(redes) > 1:
    top = max(int(v.get("n") or 0) for v in redes.values())
    print(f"\nLa red mas activa acumula {top} de {sesiones} sesiones "
          f"({100 * top / sesiones:.0f} %).")
