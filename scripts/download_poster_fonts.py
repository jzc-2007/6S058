from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


FONT_DIR = Path("/usr/local/share/fonts/poster_art")

FONTS = {
    "BebasNeue-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Anton-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Oswald.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald[wght].ttf",
    "Montserrat.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat[wght].ttf",
    "Raleway.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/raleway/Raleway[wght].ttf",
    "PlayfairDisplay.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
    "CormorantGaramond-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond-Regular.ttf",
    "CormorantGaramond-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond-Bold.ttf",
    "LibreBaskerville-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Regular.ttf",
    "LibreBaskerville-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Bold.ttf",
    "Cinzel.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzel/Cinzel[wght].ttf",
    "CinzelDecorative-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzeldecorative/CinzelDecorative-Bold.ttf",
    "Pacifico-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/pacifico/Pacifico-Regular.ttf",
    "Lobster-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lobster/Lobster-Regular.ttf",
    "Bangers-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/bangers/Bangers-Regular.ttf",
    "LuckiestGuy-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/luckiestguy/LuckiestGuy-Regular.ttf",
    "PermanentMarker-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/permanentmarker/PermanentMarker-Regular.ttf",
    "PatrickHand-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    "Orbitron.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/orbitron/Orbitron[wght].ttf",
    "Audiowide-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/audiowide/Audiowide-Regular.ttf",
    "SpaceGrotesk.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/spacegrotesk/SpaceGrotesk[wght].ttf",
    "RobotoCondensed.ttf": "https://raw.githubusercontent.com/google/fonts/main/apache/robotocondensed/RobotoCondensed[wght].ttf",
    "Fraunces.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/fraunces/Fraunces[SOFT,WONK,opsz,wght].ttf",
}


def _safe_url(url: str) -> str:
    # urllib does not accept raw square brackets in URLs.
    return url.replace("[", quote("[")).replace("]", quote("]"))


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else FONT_DIR
    target.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name, url in FONTS.items():
        out = target / name
        if out.exists() and out.stat().st_size > 10000:
            print(f"[skip] {out}")
            ok += 1
            continue
        try:
            with urlopen(_safe_url(url), timeout=60) as response:
                data = response.read()
            if len(data) < 10000:
                raise RuntimeError(f"downloaded file too small: {len(data)}")
            out.write_bytes(data)
            print(f"[font] {out} {len(data)} bytes")
            ok += 1
        except (HTTPError, URLError, RuntimeError, TimeoutError) as exc:
            print(f"[warn] failed {name}: {exc}")
    print(f"[done] downloaded/available {ok}/{len(FONTS)} fonts in {target}")


if __name__ == "__main__":
    main()
