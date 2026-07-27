---
title: C++ idioms
url: https://conformance.test/cpp-idioms
---
# C++ idioms

## Move semantics

Call `std::move` to cast an lvalue to an rvalue reference. The `std::move`
helper performs no move by itself; it only enables overload resolution to
select a move constructor.
