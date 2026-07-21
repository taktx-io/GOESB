from fastapi import APIRouter

router = APIRouter(prefix="/benchmark", tags=["benchmarks"])


@router.post("")
def submit_benchmark():
    """Submit a signed benchmark result for verification & ranking (stub)."""
    return {"accepted": False, "reason": "stub"}


@router.get("/{benchmark_id}")
def get_benchmark(benchmark_id: str):
    return {"id": benchmark_id}
