# Licensing and source availability

This document describes how Jasna's source, release packages, models, and
optional supporter components are distributed. It does not replace the
license notices in individual files or grant additional permissions.

## Public application source

Unless an individual file states otherwise, the source code in this
repository is licensed under the GNU Affero General Public License version
3.0 (`AGPL-3.0-only`). Files derived from other projects retain their
copyright and license notices.

Jasna includes code and model weights from
[Lada](https://codeberg.org/ladaapp/lada). Those components remain covered by
their applicable AGPL-3.0 notices. The vendored MMagic subset retains its
OpenMMLab copyright and Apache-2.0 notices. The LTX restoration code ported
from Lightricks LTX-2.5 (`jasna/ltx/transformer.py`, `jasna/models/ltx_vae/`)
and the LTX restoration model weights are covered by the LTX-2.x Community
License.

The complete Jasna license is in [LICENSE](LICENSE). Copyright and attribution
notices are in [NOTICE](NOTICE),
[assets/THIRD_PARTY_LICENSES.md](assets/THIRD_PARTY_LICENSES.md), and
[assets/THIRD_PARTY_MODELS.md](assets/THIRD_PARTY_MODELS.md).

## Official release packages

Official packages include the files above, the full license texts in
`licenses/`, and the release-specific source correspondence in
[RELEASE_SOURCES.md](RELEASE_SOURCES.md). The release page also provides a
SHA-256 file for every split archive.

The source tag and release package must have the same version. The tag is the
preferred form for modifying Jasna's public application code. Exact custom
dependency revisions, FFmpeg builds, patches, model hashes, and source links
are recorded in `RELEASE_SOURCES.md` and the two third-party notice files.

## Optional supporter components

Official release packages may include a separately maintained,
source-unavailable protection component. It performs local supporter-key
validation and in-memory decryption for supporter-only models. It contains no
telemetry, persistence, or network communication.

The protection component is compiled into the official executable. There is
no supported way to remove it from a prebuilt package. In the free CLI path,
when no license arguments or supporter models are selected, Jasna does not
import or call it. A public-source installation can run the free restoration
workflows without the component.

The protection implementation is proprietary and its source is not included
in the public repository. Consequently, the public source does not reproduce
the official supporter-enabled executable bit for bit. The public repository
must not be represented as complete corresponding source for that private
component.

## Models

Every bundled or downloadable model has separate artifact-specific terms and
a SHA-256 entry in `assets/THIRD_PARTY_MODELS.md`. Jasna-trained RF-DETR
detectors are licensed under Apache-2.0. Lada models retain AGPL-3.0. Encrypted
supporter models are proprietary and may be used only under their stated
supporter-model terms.

