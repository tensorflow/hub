# Common Signatures for Modules

## Introduction

Modules for the same task should implement a common signature, so that module
consumers can easily exchange them and find the best one for their problem.

This directory collects specifications of common signatures. We expect it
to grow over time, as modules are created for a wider variety of tasks.

In the best case, the specification of a common signature provides strong enough
guarantees such that consumers can call just `output = module(inputs)` without
knowing anything about the module's internals. If some adaptation is
unavoidable, we propose to supply library functions to encapsulate it, and
document them along the signature.

In any case, the goal is to make exchanging different modules for the same task
as simple as switching a string-valued hyperparameter.


## Signatures

*   [Image Signatures](images.md)
*   [Text Signatures](text.md)
