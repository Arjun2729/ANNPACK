# Public demo signing fixture

This directory contains intentionally public Ed25519 test key material used only
to make the checked-in browser demo signature byte-reproducible.

The seed is not secret. Anyone can produce signatures with it. A signature made
with this fixture proves that ANNPack signature verification works and that the
artifact was not altered after signing; it does **not** establish a publisher,
organization, domain, Google identity, or production trust relationship.

Never copy this key into a real publishing workflow. Production publishers must
use separately protected key material and an external policy that binds the
trusted public key to the claimed publisher identity.
