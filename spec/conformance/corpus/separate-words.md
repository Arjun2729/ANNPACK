---
title: Separate words
url: https://conformance.test/separate-words
---
# Separate words

## Standard library and motion

This page mentions std and move as separate words, repeatedly: std, move, std,
move, std, move. It also mentions foo and bar as separate words: foo, bar, foo,
bar, foo, bar. A reader whose tokenizer splits technical identifiers on `:` or
`_` will incorrectly match this page for those identifiers.
