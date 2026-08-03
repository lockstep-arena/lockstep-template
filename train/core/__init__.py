"""Game-agnostic training core.

Everything in here is parameterized by a discovered GameSpec (see
:mod:`train.core.discovery`) and the env's own Gymnasium spaces. Importing a
game package from this package is forbidden — games are reached ONLY through
entry-point discovery, which is what lets one template train every installed
game without ever naming one.
"""
