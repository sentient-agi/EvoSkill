# Load an env file WITHOUT overriding what the caller already exported.
#
# .env.example's first line promises "Scripts read this file when the variables are not
# already exported". Both leg launchers did the opposite: `. "$ENVF"` runs the file's
# assignments unconditionally, so an exported value loses to whatever the file holds --
# and the shipped template ships OPENROUTER_API_KEY= with an EMPTY value, so sourcing it
# actively BLANKED a key the operator had exported, after which the launcher refused with
# "OPENROUTER_API_KEY not set". It also silently defeated the standing "export the org key
# in every launcher" directive, and made EVOSKILL_REQUIRE_KEY_SUFFIX a fence around a key
# the run did not end up spending.
#
# This reads the file line by line and exports only names that are currently unset or
# empty, which is what the documentation always claimed. Deliberately conservative about
# what it will parse: KEY=VALUE with a shell-safe name, optional surrounding quotes, no
# expansion, no `export ` prefix handling beyond stripping it. Anything else is skipped
# rather than guessed at -- a .env this cannot parse is a config error the operator should
# see, not something to interpret creatively.
evoskill_load_env () {
  _ef="$1"
  [ -f "$_ef" ] || return 0
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in
      ''|'#'*) continue ;;
      'export '*) _line=${_line#export } ;;
    esac
    case "$_line" in *=*) ;; *) continue ;; esac
    _k=${_line%%=*}
    _v=${_line#*=}
    # strip surrounding whitespace on the name, reject anything not a shell identifier
    _k=$(printf '%s' "$_k" | tr -d ' \t')
    case "$_k" in
      ''|*[!A-Za-z0-9_]*) continue ;;
      [0-9]*) continue ;;
    esac
    # EXPORTED WINS: skip if already set to a non-empty value
    eval "_cur=\${$_k:-}"
    [ -n "$_cur" ] && continue
    # strip one layer of matching quotes
    case "$_v" in
      \"*\") _v=${_v#\"}; _v=${_v%\"} ;;
      \'*\') _v=${_v#\'}; _v=${_v%\'} ;;
    esac
    export "$_k=$_v"
  done < "$_ef"
  unset _ef _line _k _v _cur
}
