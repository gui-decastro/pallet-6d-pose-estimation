#!/usr/bin/env bash
# Strips snap library paths from LD_LIBRARY_PATH before exec-ing the target.
#
# Problem: VS Code is installed as a snap, so its integrated terminal inherits
# snap's environment. This causes snap's core20 libpthread (Ubuntu 20.04) to
# shadow the system one, producing:
#   "symbol lookup error: /snap/core20/.../libpthread.so.0:
#    undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE"
#
# Fix: filter /snap/* entries out of LD_LIBRARY_PATH and unset GTK_PATH
# (GTK_PATH pointing into snap triggers the snap lib load chain).

export LD_LIBRARY_PATH=$(
    printf '%s' "${LD_LIBRARY_PATH:-}" \
    | tr ':' '\n' \
    | grep -v '^/snap/' \
    | tr '\n' ':' \
    | sed 's/:$//'
)
unset GTK_PATH

exec "$@"
