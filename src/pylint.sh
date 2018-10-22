#!/bin/bash
if [[ "${MAKEFLAGS}" =~ --jobserver-fds=([0-9]*),([0-9]*) ]]
then
  jobslots=$(timeout 0.01 cat <&${BASH_REMATCH[1]})
  echo pylint -j${#jobslots} "$@"
  pylint -j${#jobslots} "$@"
  echo -n "${jobslots}" >&${BASH_REMATCH[2]}
else
  echo pylint "$@"
  exec pylint "$@"
fi
