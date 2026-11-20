# Source attribution

## Project authorship and release scope

The Riemann-RSI-Preview authors confirm that the project's own inference implementation
and project-specific adaptations are original work without a separate public
upstream. That code is supplied in this submission. This authorship declaration
does not relabel the retained third-party code, models or resources listed below.

This release is eval-only. Training code is not being open-sourced, no public
release date is currently announced, and there is no paper or arXiv preprint.

This records retained component origins. The author authorized publication of the
seven learned-policy files as Riemann1.0 in `bolwang/Riemann-RSI-Preview_robodojo`. Weight and
normalization hashes are preserved; the inference-only TOML has a new digest and
omits private training configuration. Retained third-party attribution is
recorded separately below.

## Verified publication and retained components

| Component | Evidence available | Scope |
|---|---|---|
| Riemann1.0 | Public HF revision `e0fa3f2a920eb74ae105194829e13cb07b861c08`; seven file sizes/digests match the current download declaration. Weight and normalization identities are unchanged. | The model card declares Apache-2.0. Training code/data are not included in this eval-only release; retained third-party components keep their own terms. |
| SkyRobot inference | `WanRotaryPosEmbed` matches Lingbot-VA revision `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`; its Apache-2.0 license and scoped [source notice](../policies/riemann_1/implementation/skyrobot/SOURCE_NOTICE.md) are included. Supplied Robbyant utilities have documented public counterparts and local adaptations; Wan T5/VAE files retain Alibaba Wan copyright headers. | Project-authored inference and adaptations are original per the authors' declaration above. The named third-party portions retain their separate attribution. |
| Alibaba Wan inference | Wan2.2 revision `1ea34ff48f87168174e12956e200b1d908b1c5ff`: tokenizer bytes match exactly; 14/15 T5 and 15/18 VAE top-level definitions have identical syntax trees. The [source notice](../policies/riemann_1/implementation/skyrobot/SOURCE_NOTICE.md) describes every changed definition and links its pinned source. The retained Apache-2.0 text matches upstream byte-for-byte. | Fixed-length text context, memory cleanup, streaming/batched VAE and autocast adaptations are supplied integration changes. This verifies those three files' source relationships, not the remaining SkyRobot tree, checkpoint or training data. |
| Rule distributions and small resources | Original Fold v45 / Pour v21 distributions supplied the MIT notice retained below; task composition files identify current resource inputs. | The supplied notices and task-specific source attribution are retained. Numerical resource identity is separate from authorship. |
| Wan encoder / VAE / tokenizer | Six unchanged upstream files mirrored together with the common models; pinned URLs and hashes in `configs/models/downloads.json`. | Preserve their upstream terms and notices independently of the Riemann1.0 model card. |
| SAM3 / CLIP / GroundingDINO | Exact mirrors and original notices are under `common_visual_models/`; SAM runtime dependency notices accompany the small source/runtime archive. | Their individual terms apply, independently of the policy model. |
| Task detectors / templates / protocol | Two classifiers are under `task_detectors/`; templates are deduplicated into one hash-addressed cache. `download_checkpoints.sh --prepare` installs them and reconstructs the admitted WebSocket code from its fixed public wheel. | These are supplied mature task resources. The public mirror and content identities identify the evaluated inputs; this integration does not claim a new detector-training release or new performance certification. |

## Retained components

- Small public own-robot and garment template resources are grouped under `assets/robots/` and `assets/perception/`. The supplied MIT notice is consolidated below; numerical resource identity is retained separately from filename/location.

- `policies/riemann_1/adapter/` and the retained `implementation/skyrobot/`
  inference modules originate from the supplied WAM integration. The exact seven
  checkpoint/configuration/normalization files are recorded in `configs/models/downloads.json`.
  The model release route is [Riemann1.0](https://huggingface.co/bolwang/Riemann-RSI-Preview_robodojo).
  The Riemann1.0 subfolder retains its declared model license; the repository card lists mixed component terms.
- The retained SkyRobot inference includes the geometry utility
  `skyrobot/train/utils.py`. A directory name is not a license or
  grounds for deleting an inference dependency. Original in-file notices remain
  unmodified. The verified Robbyant portions have a scoped source notice and the
  unmodified upstream Apache-2.0 license. Project-owned portions are covered by
  the authorship declaration above; identified reused portions keep their notices.
- The Wan encoder and SAM resources retain their declared public author identities.
  Their terms do not license the separate Riemann1.0 checkpoint or its training data.
- Python libraries are installed from their package distributions, not vendored into
  this source archive. The installed environments retain their working distributions,
  with their own license metadata. Preserve those distributions' notices.

## Submission status

The PR declares the eval-only scope and the authors' original-code statement.
Third-party notices remain separate from that statement. Local validation and
simulator results do not imply maintainer agreement or official-cloud acceptance.

## Original MIT notice

Supplied with the original Fold v45 (`source/embedded/RoboDojo-LICENSE`) and
Pour v21 (`source/embedded/policy/LICENSE-RoboDojo`) distributions.

```text
MIT License

Copyright (c) 2025 Yue Chen <yuechen020614@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Language-directed pouring imports the public-image R8 Rule candidate, with
its numerical planning/vision/motion attribution retained in
[`SOURCE_NOTICE.md`](../tasks/pour_by_language/SOURCE_NOTICE.md) and the
[Apache-2.0 license](../tasks/pour_by_language/SOURCE_LICENSE.txt).
The donor pilots are not scores for this integrated release.
