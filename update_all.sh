#!/usr/bin/env bash

custom_nodes="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/custom_nodes"
failed=()

for node in "$custom_nodes"/*/; do
    [ -d "$node/.git" ] || continue
    name="$(basename "$node")"

    echo "== $name"

    if ! (
        cd "$node" || exit 1

        git fetch --all --prune --tags

        upstream="$(git symbolic-ref -q refs/remotes/origin/HEAD)" || upstream="refs/remotes/origin/main"

        git rebase "${upstream#refs/remotes/}"
    ); then
        failed+=("$name")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    joined=$(printf '%s, ' "${failed[@]}")
    echo "Failed: ${joined%, }"
fi
