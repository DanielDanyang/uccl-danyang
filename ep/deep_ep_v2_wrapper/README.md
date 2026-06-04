# DeepEP V2 Wrapper for AWS EFA

This wrapper installs a `deep_ep` Python package that exposes a DeepEP
V2-compatible API while routing AWS EFA communication through the UCCL-style
proxy backend.

This wrapper targets the DeepEP V2 `ElasticBuffer` API.

## Intended install flow

```bash
cd uccl-ep
python setup.py install

cd deep_ep_v2_wrapper
python setup.py install
```

The first command installs the native `uccl.ep` extension. The second command
installs a `deep_ep` package that exposes `ElasticBuffer` and V2 handle objects.

## Porting notes

The implementation is intentionally narrow:

- AWS EFA only.
- EP16 / p5en first.
- `ElasticBuffer.dispatch` and `ElasticBuffer.combine` first.
- V2 metadata, expanded-dispatch scatter, and reduced-combine gather are native
  CUDA helpers on `ElasticProxyBuffer`.
- The old DeepEP V1 `deep_ep_wrapper` package has been removed from this
  workspace; remaining proxy transport compatibility is confined to the
  internode data path until the V2 transfer plan replaces it.
- Other DeepEP V2 APIs should fail explicitly until implemented.
