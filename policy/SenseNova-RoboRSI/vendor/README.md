# Vendored RoboDojo source

`roborsi_source/` retains 213 unchanged files from RoboRSI baseline
`cfa591575172b638a8436fa95db28e9469785c9c`. Unused duplicate legacy launch data
is omitted; the native evaluator and required transport modules remain intact.

`robodojo_adaptive/` contains the separately authenticated adaptive snapshot:
21 source files, 42 task configurations, 42 public interfaces and one mapping.
Deployment paths and standalone wrappers are portable; policy parameters,
budgets, controller algorithms and the sandbox function are preserved.

`pi05/` retains its 197 service/OpenPI/dependency files and licenses. One
conversion-script documentation example uses relative paths.

Each source manifest records actual SHA-256 digests and, where applicable,
upstream identities, path-only patches or omitted files. Copyright notices and
licenses are retained. The fixed 83-package tool archive, model weights and 27
ICL payloads are separate, unchanged assets.
