from fastapi import APIRouter, Depends

router = APIRouter(prefix="/utils", tags=["utils"])

@router.get("/health-check/")
async def health_check() -> bool:
    return True
