# deepseek_pin_guard.sh — #319: a DeepSeek-family model dispatched through the CC harness
# MUST route through a local proxy that injects the native-provider pin.
#
# WHY THIS EXISTS. The `claude` CLI builds its own request bodies and has no field for
# OpenRouter provider preferences, so a direct-to-OpenRouter dispatch is served by
# whatever host the router picks PER REQUEST. ssbv2_selfevo's e1/e2 evolver sessions ran
# this way and were served by Baidu, Amazon Bedrock, StreamLake and Io Net — zero sampled
# generations from DeepSeek native — and two other hosts (DigitalOcean, Sail Research)
# produced visible protocol corruption when e3 landed on them. The pin rule
# (order=["deepseek"] + allow_fallbacks=false, DeepSeek family only) lives in
# scripts/cc_proxy.py; this guard makes it impossible to dispatch AROUND it.
#
# ONE GUARD, HERE. Do not copy this check into Python callers — two guards, one real,
# is the #154 shape. Callers source this file and call require_deepseek_native_route.

require_deepseek_native_route() {
  local model="$1" base="${ANTHROPIC_BASE_URL:-}"
  case "$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')" in
    deepseek/*) ;;
    *) return 0 ;;   # not DeepSeek-family: no pin requirement on this route
  esac
  case "$base" in
    http://127.0.0.1:*|http://localhost:*) ;;
    "")
      echo "REFUSING: $model via the CC harness with ANTHROPIC_BASE_URL unset — claude would" >&2
      echo "  call Anthropic with an OpenRouter token (the e1 auth404) or route unpinned." >&2
      echo "  Start scripts/cc_proxy.py and export ANTHROPIC_BASE_URL=http://127.0.0.1:8100." >&2
      return 4 ;;
    *)
      echo "REFUSING: $model via the CC harness with ANTHROPIC_BASE_URL=$base — a non-local" >&2
      echo "  route cannot inject the native-provider pin. ssbv2_selfevo e1/e2 ran this way" >&2
      echo "  and were served by 4 third-party hosts (#319). Route through scripts/cc_proxy.py." >&2
      return 4 ;;
  esac
  # A live port is not a pin. The proxy must DECLARE the DeepSeek pin rule itself, so a
  # stray local server (or an old cc_proxy build without #314) cannot satisfy this guard.
  local pin_info
  pin_info=$(curl -sf --max-time 5 "$base/pin" 2>/dev/null) || {
    echo "REFUSING: nothing answering at $base/pin — start scripts/cc_proxy.py first." >&2
    return 4; }
  case "$pin_info" in
    *'"deepseek_pin": true'*|*'"deepseek_pin":true'*) ;;
    *)
      echo "REFUSING: the server at $base does not declare deepseek_pin:true — it would" >&2
      echo "  forward unpinned. Got: $(printf '%.120s' "$pin_info")" >&2
      return 4 ;;
  esac
  echo "  [PIN-GUARD] $model -> $base (proxy declares the native DeepSeek pin)" >&2
  return 0
}
