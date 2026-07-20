# ANN-5: Policy descriptors

Status: implemented metadata draft. Requires ANNPack Core v1.0-draft.

ANN-5 adds declarative payment and encryption descriptors to the manifest policy object. It communicates acquisition and handling requirements; it does not implement payment settlement, authorization, key delivery, encrypted section codecs, DRM, or enforcement after plaintext access.

A producer must not advertise payment or encryption as enforced merely because ANN-5 metadata exists. Consumers that do not implement ANN-5 may expose the metadata opaquely or ignore it.
