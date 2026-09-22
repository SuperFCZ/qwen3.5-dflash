"""Dedicated control route using vLLM's existing worker collective RPC."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.post("/eqc_cuda_profile/{action}")
async def cuda_event_profile(action: str, request: Request):
    if action not in {"start", "stop", "snapshot"}:
        raise HTTPException(status_code=400, detail="expected start, stop, or snapshot")
    engine = request.app.state.engine_client
    if action in {"start", "stop"}:
        # Let scheduled/in-flight execute+sample pairs finish before resetting or
        # synchronizing. Use a dedicated benchmark server with no other clients.
        await engine.wait_for_requests_to_drain()
    records = await engine.collective_rpc("eqc_cuda_event_profile", timeout=60.0, args=(action,))
    return {"workers": records}


def install_api_route() -> None:
    # build_app imports this factory at call time in vLLM 0.22.1.
    import vllm.entrypoints.serve as serve

    original = serve.register_vllm_serve_api_routers
    if getattr(original, "_eqc_profile_installed", False):
        return

    def register(app):
        original(app)
        app.include_router(router)

    register._eqc_profile_installed = True
    serve.register_vllm_serve_api_routers = register
