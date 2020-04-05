function prepare() {
  script_dir="$(realpath "$(dirname $0)")"
  script_name="$(basename $0)"
  script_name="${script_name/%.sh/}"
  config="${script_dir}/env.sh"
  base_dir="${script_dir}/.."
  log_dir="${script_dir}/log"
  mkdir -p "${log_dir}"
  tmp_dir="$(mktemp /tmp/soda-${script_name}.XXXXXXXX --directory)"
  pushd "${tmp_dir}" >/dev/null

  log_file="${log_dir}/$(date +%y%m%d-%H%M%S).log"
  exec {log_fd}>${log_file}
  ln --symbolic --force "$(basename "${log_file}")" \
    "${log_dir}/${script_name}.log"

  test -f "${config}" && . "${config}"
}

function log() {
  echo "$@"
  if [ "$1" = "-n" ]; then
    shift
  fi
  echo "$@" >&2
}

function cleanup() {
  popd >/dev/null
  rm -rf "${tmp_dir}"
}