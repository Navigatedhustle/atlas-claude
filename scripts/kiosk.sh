#!/bin/bash
# Open Atlas full screen in its own browser profile (never touches your normal profile).
# Autoplay is allowed so attract mode can speak before anyone clicks; mic/camera permission is pre-granted.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# When launched from SSH/tmux, find the local desktop display (this box uses :1).
if [ -z "${DISPLAY:-}" ]; then
  for x in /tmp/.X11-unix/X*; do [ -e "$x" ] && export DISPLAY=":${x##*/X}" && break; done
fi
PROFILE="$ROOT/run/browser-profile"; mkdir -p "$PROFILE"
BROWSER=$(command -v google-chrome || command -v chromium || command -v chromium-browser || echo /snap/bin/chromium)
exec "$BROWSER" --user-data-dir="$PROFILE" --kiosk --no-first-run --noerrdialogs --disable-translate \
  --autoplay-policy=no-user-gesture-required --use-fake-ui-for-media-stream \
  --disable-background-networking --disable-component-update --disable-sync --disable-domain-reliability \
  --disable-client-side-phishing-detection --no-default-browser-check --no-pings \
  --disable-features=Translate,MediaRouter,OptimizationHints "$@" http://localhost:8100
