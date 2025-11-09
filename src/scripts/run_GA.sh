#!/usr/bin/env bash
set -euo pipefail

# ===== Defaults (podem ser sobrescritos por env e/ou CLI) =====
PATH_ORLIB="${PATH_ORLIB:-../../instances/mkp/mknapcb1.txt}"
CLASS="${CLASS:-problems.mkp.solvers.GA_MKP}"
CP="${CP:-../../out}"

POP="${POP:-100}"
MUT="${MUT:-0.02}"
REPAIR="${REPAIR:-true}"
LAMBDA="${LAMBDA-}"       # vazio => lambda automático

TS="${TS:-false}"
TENURE="${TENURE-}"
STEPS="${STEPS-}"
VMIN="${VMIN-}"
VMAX="${VMAX-}"
LMBMIN="${LMBMIN-}"
LMBMAX="${LMBMAX-}"
UP="${UP-}"
DOWN="${DOWN-}"

RANGE_START="${RANGE_START:-1}"
RANGE_END="${RANGE_END:-10}"

# ===== Parse CLI (sobrescreve os defaults acima) =====
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --path) PATH_ORLIB="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --cp) CP="$2"; shift 2 ;;

    --pop|--popSize) POP="$2"; shift 2 ;;
    --mutation|--mut) MUT="$2"; shift 2 ;;
    --repair) REPAIR="$2"; shift 2 ;;
    --lambda|--lmbda|--lmb) LAMBDA="$2"; shift 2 ;;

    --ts) TS="$2"; shift 2 ;;
    --tenure) TENURE="$2"; shift 2 ;;
    --ts-steps|--steps) STEPS="$2"; shift 2 ;;
    --vmin) VMIN="$2"; shift 2 ;;
    --vmax) VMAX="$2"; shift 2 ;;
    --lmbMin) LMBMIN="$2"; shift 2 ;;
    --lmbMax) LMBMAX="$2"; shift 2 ;;
    --up) UP="$2"; shift 2 ;;
    --down) DOWN="$2"; shift 2 ;;

    --range-start) RANGE_START="$2"; shift 2 ;;
    --range-end) RANGE_END="$2"; shift 2 ;;

    --) shift; break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;   # passa adiante qualquer flag desconhecida
  esac
done

# ===== Montagem de argumentos =====
BASENAME="$(basename "$PATH_ORLIB" .txt)"
COMMON_ARGS=( --path "$PATH_ORLIB" --pop "$POP" --mutation "$MUT" --repair "$REPAIR" )
[[ -n "$LAMBDA" ]] && COMMON_ARGS+=( --lambda "$LAMBDA" )

# Detecta se TS deve ligar
ts_on=false
if [[ "$TS" == "true" ]]; then
  ts_on=true
elif [[ -n "${TENURE}" || -n "${STEPS}" || -n "${VMIN}" || -n "${VMAX}" || -n "${LMBMIN}" || -n "${LMBMAX}" || -n "${UP}" || -n "${DOWN}" ]]; then
  ts_on=true
fi

if $ts_on; then
  COMMON_ARGS+=( --ts true )
  [[ -n "$TENURE" ]] && COMMON_ARGS+=( --tenure "$TENURE" )
  [[ -n "$STEPS"  ]] && COMMON_ARGS+=( --ts-steps "$STEPS" )
  [[ -n "$VMIN"   ]] && COMMON_ARGS+=( --vmin "$VMIN" )
  [[ -n "$VMAX"   ]] && COMMON_ARGS+=( --vmax "$VMAX" )
  [[ -n "$LMBMIN" ]] && COMMON_ARGS+=( --lmbMin "$LMBMIN" )
  [[ -n "$LMBMAX" ]] && COMMON_ARGS+=( --lmbMax "$LMBMAX" )
  [[ -n "$UP"     ]] && COMMON_ARGS+=( --up "$UP" )
  [[ -n "$DOWN"   ]] && COMMON_ARGS+=( --down "$DOWN" )
fi

FINAL_ARGS=( "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" )

# ===== Logs =====
suffix="GA_pop${POP}_mut${MUT}_repair${REPAIR}"
if $ts_on; then
  suffix+="_ts"
  [[ -n "$TENURE" ]] && suffix+="_tenure${TENURE}"
  [[ -n "$STEPS"  ]] && suffix+="_steps${STEPS}"
  [[ -n "$VMIN"   ]] && suffix+="_vmin${VMIN}"
  [[ -n "$VMAX"   ]] && suffix+="_vmax${VMAX}"
  [[ -n "$LMBMIN" ]] && suffix+="_lmbMin${LMBMIN}"
  [[ -n "$LMBMAX" ]] && suffix+="_lmbMax${LMBMAX}"
  [[ -n "$UP"     ]] && suffix+="_up${UP}"
  [[ -n "$DOWN"   ]] && suffix+="_down${DOWN}"
fi
LOGDIR="${LOGDIR:-runs/${BASENAME}/${suffix}}"
mkdir -p "$LOGDIR"
COMMON_ARGS+=( --results-dir "${LOGDIR}" --variant "${suffix}"  --mkcbres "../../instances/mkp/mkcbres.txt" --algo "GA" )

echo "Rodando ${CLASS} para ${PATH_ORLIB} [instâncias ${RANGE_START}-${RANGE_END}]"
echo "Logs: ${LOGDIR}"
echo "TS ativado? $ts_on"

# Salva a linha exata de execução (com aspas corretas)
{ printf 'java -cp %q %q --instance <ID> ' "$CP" "$CLASS"; printf '%q ' "${FINAL_ARGS[@]}"; echo; } > "${LOGDIR}/run.args"

# ===== Execução paralela =====
seq "$RANGE_START" "$RANGE_END" | xargs -n 1 -I{} bash -c '
  inst="$1"; shift
  logdir="$1"; shift
  cp="$1"; shift
  class="$1"; shift
  echo "[$(date +%H:%M:%S)] Instância ${inst} iniciada..."
  java -cp "$cp" "$class" --instance "$inst" "$@" > "${logdir}/inst_${inst}.log" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] Instância ${inst} finalizada (OK). Log: ${logdir}/inst_${inst}.log"
  else
    echo "[$(date +%H:%M:%S)] Instância ${inst} falhou (status ${status}). Ver: ${logdir}/inst_${inst}.log" >&2
  fi
' _ {} "$LOGDIR" "$CP" "$CLASS" "${FINAL_ARGS[@]}"
