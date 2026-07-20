# ANN-6: Pack dependencies

Status: implemented draft. Requires ANNPack Core v1.0-draft.

ANN-6 lets a manifest identify another pack by name, version requirement, optional immutable root, and optional discovery URL. A consumer must retain the originating pack identity on every result and must not silently merge conflicting versions.

ANN-6 does not define recursive resolution policy, a federation query planner, dependency installation, lockfiles, or conflict arbitration. Those remain application decisions and are not implied by the existence of dependency records.
