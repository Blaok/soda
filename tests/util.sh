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

  log_file="${log_dir}/${script_name}.$(date +%y%m%d-%H%M%S).log"
  exec {log_fd}>${log_file}
  ln --symbolic --force "$(basename "${log_file}")" \
    "${log_dir}/${script_name}.log"

  test -f "${config}" && . "${config}"

  exec {stderr_fd}<&2
  trap "cat ${log_dir}/${script_name}.log >&${stderr_fd}" ERR
}

function log() {
  echo -n "$@"
  echo "$@" >&2
}

function pass() {
  echo -e "\x1B[0;32mPASS\x1B[0m"
  echo -e "PASS\n" >&2
}

function fail() {
  echo -e "\x1B[0;31mFAIL\x1B[0m"
  echo -e "FAIL\n" >&2
  return 1
}

function cleanup() {
  popd >/dev/null
  rm -rf "${tmp_dir}"
}

set -e
