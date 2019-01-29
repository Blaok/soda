#!/bin/sh
tmpdir=$(mktemp -d /tmp/git-pre-commit-tmpdir-XXXXXXXX)
function finish {
  rm -rf "${tmpdir}"
}
trap finish EXIT
set -e
git checkout-index --prefix=${tmpdir}/ -af
git submodule foreach "git checkout-index --prefix=${tmpdir}/\${name}/ -af"
git diff-index --cached --name-only --diff-filter=D -z HEAD | \
  (cd ${tmpdir} && xargs -0 rm -f --)
cd ${tmpdir}
make pre-commit -j$(nproc)
