# Licensing and source availability

This document describes how Jasna's public source and optional supporter
components are distributed. It does not replace the license notices in
individual files or grant additional permissions.

## Public source

Unless an individual file states otherwise, the source code in this repository
is licensed under the GNU Affero General Public License version 3.0
(`AGPL-3.0`). Files derived from other projects retain their copyright and
license notices.

Jasna includes code and model weights from
Lada. Those components remain covered by
their applicable AGPL-3.0 notices. The vendored MMagic subset retains its
OpenMMLab copyright and license notices.

A clean checkout of this repository can be installed and used for Jasna's free
restoration workflows without the optional `jasna.protection` package.

## Optional supporter components

Official release packages may additionally include a separately developed,
source-unavailable protection component. It performs local supporter-key
validation for the supporter-only `unet-4x` and
SD 1.5 restoration models. When that component is unavailable, supporter-only
models are unavailable, while the free restoration workflows remain usable.

The protection component is maintained as a separate project, can be reused by
other applications, and is distributed under separate proprietary terms. It
is not included in the public Jasna source repository.

The protection component contains debugger and core-dump resistance intended
to restrict access to supporter-only models. It does not
implement telemetry, persistence, or network communication.

## Release reproducibility

The public repository contains the application source and development setup
for the free restoration workflows. It does not contain the proprietary
protection component or the maintainer's standalone release tooling.
Consequently, a public checkout does not reproduce a supporter-enabled
official release package bit for bit.

